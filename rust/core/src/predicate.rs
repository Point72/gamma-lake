//! Predicates in disjunctive normal form, translated to DataFusion expressions.

use arrow::array::{Array, ArrayRef, BooleanArray, Scalar};
use arrow::compute::kernels::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow::compute::{
    and_kleene, cast_with_options, filter_record_batch, is_not_null, is_null, or_kleene,
    CastOptions,
};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::{Column, ScalarValue, TableReference};
use datafusion::logical_expr::utils::{conjunction, disjunction};
use datafusion::prelude::{lit, Expr};

use crate::error::{Error, Result};

/// Comparison operator of a [`Term`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    /// True when the value is non-null and equals a member of `values`.
    In,
    /// True when the value is non-null and equals no member of `values`.
    NotIn,
    IsNull,
    IsNotNull,
}

impl std::str::FromStr for Op {
    type Err = Error;

    /// Parses `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not_in`, `is_null` or `is_not_null`.
    fn from_str(op: &str) -> Result<Self> {
        Ok(match op {
            "==" => Op::Eq,
            "!=" => Op::NotEq,
            "<" => Op::Lt,
            "<=" => Op::LtEq,
            ">" => Op::Gt,
            ">=" => Op::GtEq,
            "in" => Op::In,
            "not_in" => Op::NotIn,
            "is_null" => Op::IsNull,
            "is_not_null" => Op::IsNotNull,
            other => {
                return Err(Error::InvalidPredicate(format!(
                    "unknown operator {other:?}"
                )))
            }
        })
    }
}

/// One comparison between a column and literal values.
///
/// Comparison operators take exactly one non-null literal, `In`/`NotIn` take any number of
/// non-null literals, and `IsNull`/`IsNotNull` ignore `values`. Literals are cast to the
/// column type; a cast that is lossy or crosses type families is rejected.
#[derive(Clone, Debug)]
pub struct Term {
    pub column: String,
    pub op: Op,
    pub values: ArrayRef,
}

/// OR of ANDs of [`Term`]s. An empty outer list matches every row.
#[derive(Clone, Debug, Default)]
pub struct Dnf(pub Vec<Vec<Term>>);

impl Dnf {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn columns(&self) -> impl Iterator<Item = &str> {
        self.0.iter().flatten().map(|t| t.column.as_str())
    }

    /// Builds the filter expression against `schema`, qualifying columns with `relation`.
    #[doc(hidden)]
    pub fn to_expr(&self, schema: &Schema, relation: &str) -> Result<Option<Expr>> {
        let all = self
            .0
            .iter()
            .map(|terms| {
                let terms = terms.iter().map(|t| term_expr(t, schema, relation));
                Ok(conjunction(terms.collect::<Result<Vec<_>>>()?).unwrap_or_else(|| lit(true)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(disjunction(all))
    }

    /// Applies this predicate to an Arrow batch.
    pub(crate) fn filter_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let Some(mask) = self.mask(batch)? else {
            return Ok(batch.clone());
        };
        Ok(filter_record_batch(batch, &mask)?)
    }

    fn mask(&self, batch: &RecordBatch) -> Result<Option<BooleanArray>> {
        let mut any = None;
        for conjunction in &self.0 {
            let mut all = None;
            for term in conjunction {
                let mask = term_mask(batch, term)?;
                all = Some(match all {
                    Some(previous) => and_kleene(&previous, &mask)?,
                    None => mask,
                });
            }
            let all = all.unwrap_or_else(|| BooleanArray::from(vec![true; batch.num_rows()]));
            any = Some(match any {
                Some(previous) => or_kleene(&previous, &all)?,
                None => all,
            });
        }
        Ok(any)
    }
}

fn term_mask(batch: &RecordBatch, term: &Term) -> Result<BooleanArray> {
    let values = batch
        .column_by_name(&term.column)
        .ok_or_else(|| Error::InvalidPredicate(format!("unknown column {:?}", term.column)))?;
    // Empty sets need no literal type, as in `term_expr`.
    match term.op {
        Op::IsNull => return Ok(is_null(values.as_ref())?),
        Op::IsNotNull => return Ok(is_not_null(values.as_ref())?),
        Op::In if term.values.is_empty() => {
            return Ok(BooleanArray::from(vec![false; batch.num_rows()]))
        }
        Op::NotIn if term.values.is_empty() => return Ok(is_not_null(values.as_ref())?),
        _ => {}
    }
    let literals = cast_literal(&term.values, values.data_type(), &term.column)?;
    let compare = |index: usize| {
        let literal = Scalar::new(literals.slice(index, 1));
        match term.op {
            Op::Eq | Op::In => eq(values, &literal),
            Op::NotEq | Op::NotIn => neq(values, &literal),
            Op::Lt => lt(values, &literal),
            Op::LtEq => lt_eq(values, &literal),
            Op::Gt => gt(values, &literal),
            Op::GtEq => gt_eq(values, &literal),
            Op::IsNull | Op::IsNotNull => unreachable!(),
        }
    };
    match term.op {
        Op::In => {
            Ok((1..literals.len())
                .try_fold(compare(0)?, |mask, i| or_kleene(&mask, &compare(i)?))?)
        }
        Op::NotIn => {
            Ok((1..literals.len())
                .try_fold(compare(0)?, |mask, i| and_kleene(&mask, &compare(i)?))?)
        }
        op => {
            if literals.len() != 1 {
                return Err(Error::InvalidPredicate(format!(
                    "{op:?} on column {:?} needs exactly one literal, got {}",
                    term.column,
                    literals.len()
                )));
            }
            Ok(compare(0)?)
        }
    }
}

pub(crate) fn column(relation: &str, name: &str) -> Expr {
    Expr::Column(Column::new(Some(TableReference::bare(relation)), name))
}

fn term_expr(term: &Term, schema: &Schema, relation: &str) -> Result<Expr> {
    let field = schema
        .field_with_name(&term.column)
        .map_err(|_| Error::InvalidPredicate(format!("unknown column {:?}", term.column)))?;
    let col = column(relation, &term.column);
    // Empty sets need no literal type, so any (even Null-typed) empty array is accepted.
    match term.op {
        Op::IsNull => return Ok(col.is_null()),
        Op::IsNotNull => return Ok(col.is_not_null()),
        Op::In if term.values.is_empty() => return Ok(lit(false)),
        Op::NotIn if term.values.is_empty() => return Ok(col.is_not_null()),
        _ => {}
    }
    if term.values.null_count() > 0 {
        return Err(Error::InvalidPredicate(format!(
            "null literal for column {:?}; use IsNull/IsNotNull",
            term.column
        )));
    }
    let values = cast_literal(&term.values, field.data_type(), &term.column)?;
    let mut literals = (0..values.len())
        .map(|i| ScalarValue::try_from_array(&values, i).map(lit))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    match term.op {
        Op::In => Ok(col.in_list(literals, false)),
        Op::NotIn => Ok(col.in_list(literals, true)),
        op => {
            if literals.len() != 1 {
                return Err(Error::InvalidPredicate(format!(
                    "{op:?} on column {:?} needs exactly one literal, got {}",
                    term.column,
                    literals.len()
                )));
            }
            let value = literals.pop().expect("one literal");
            Ok(match op {
                Op::Eq => col.eq(value),
                Op::NotEq => col.not_eq(value),
                Op::Lt => col.lt(value),
                Op::LtEq => col.lt_eq(value),
                Op::Gt => col.gt(value),
                Op::GtEq => col.gt_eq(value),
                Op::In | Op::NotIn | Op::IsNull | Op::IsNotNull => unreachable!(),
            })
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Family {
    Numeric,
    String,
    Binary,
    Boolean,
    Date,
    /// Timestamps with a time zone denote instants; naive ones do not, so they never mix.
    Timestamp {
        zoned: bool,
    },
    List(Box<Family>),
    Struct(Vec<(String, Family)>),
    Other(DataType),
}

fn family(data_type: &DataType) -> Family {
    use DataType::*;
    match data_type {
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 | Float16 | Float32
        | Float64 | Decimal128(..) | Decimal256(..) => Family::Numeric,
        Utf8 | LargeUtf8 | Utf8View => Family::String,
        Binary | LargeBinary | BinaryView | FixedSizeBinary(_) => Family::Binary,
        Boolean => Family::Boolean,
        Date32 | Date64 => Family::Date,
        Timestamp(_, tz) => Family::Timestamp {
            zoned: tz.is_some(),
        },
        Dictionary(_, value) => family(value),
        List(f) | LargeList(f) | ListView(f) | LargeListView(f) | FixedSizeList(f, _) => {
            Family::List(Box::new(family(f.data_type())))
        }
        Struct(fields) => Family::Struct(
            fields
                .iter()
                .map(|f| (f.name().clone(), family(f.data_type())))
                .collect(),
        ),
        other => Family::Other(other.clone()),
    }
}

/// Casts `values` to `target`, rejecting casts across type families or that lose information.
fn cast_literal(values: &ArrayRef, target: &DataType, column: &str) -> Result<ArrayRef> {
    lossless_cast(values, target).map_err(|why| {
        Error::InvalidPredicate(format!(
            "literal of type {} cannot be compared with column {column:?} of type {}: {why}",
            values.data_type(),
            match target {
                DataType::Dictionary(_, value) => value.as_ref(),
                other => other,
            }
        ))
    })
}

/// Casts `values` to `target` (the value type for dictionaries) when no value changes;
/// otherwise returns why not.
pub(crate) fn lossless_cast(
    values: &ArrayRef,
    target: &DataType,
) -> std::result::Result<ArrayRef, String> {
    let target = match target {
        DataType::Dictionary(_, value) => value.as_ref(),
        other => other,
    };
    if values.data_type() == target {
        return Ok(values.clone());
    }
    let (from, to) = (family(values.data_type()), family(target));
    if from != to {
        return Err("incompatible types".into());
    }
    let strict = CastOptions {
        safe: false,
        ..Default::default()
    };
    let cast = cast_with_options(values, target, &strict).map_err(|e| e.to_string())?;
    if widens(values.data_type(), target) {
        return Ok(cast);
    }
    let back = cast_with_options(&cast, values.data_type(), &strict).map_err(|e| e.to_string())?;
    if back.to_data() != values.to_data() {
        return Err("a value is not exactly representable".into());
    }
    Ok(cast)
}

/// Whether every value of `from` is exactly representable in `to`, so a strict cast that
/// succeeds needs no round-trip check.
fn widens(from: &DataType, to: &DataType) -> bool {
    use DataType::*;
    let rank = |t: &DataType| match t {
        Int8 => Some(1),
        Int16 => Some(2),
        Int32 => Some(3),
        Int64 => Some(4),
        _ => None,
    };
    match (from, to) {
        (Utf8 | LargeUtf8 | Utf8View, Utf8 | LargeUtf8 | Utf8View) => true,
        (Binary | LargeBinary | BinaryView, Binary | LargeBinary | BinaryView) => true,
        (Dictionary(_, value), _) => value.as_ref() == to || widens(value, to),
        (Float16 | Float32, Float64) | (Float16, Float32) => true,
        _ => matches!((rank(from), rank(to)), (Some(a), Some(b)) if a <= b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray, TimestampNanosecondArray};
    use arrow::datatypes::{Field, TimeUnit};
    use std::sync::Arc;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("n", DataType::Int32, true),
            Field::new("s", DataType::Utf8View, true),
        ])
    }

    fn term(column: &str, op: Op, values: ArrayRef) -> Term {
        Term {
            column: column.into(),
            op,
            values,
        }
    }

    fn expr(terms: Vec<Vec<Term>>) -> Result<Option<Expr>> {
        Dnf(terms).to_expr(&schema(), "t")
    }

    #[test]
    fn empty_dnf_has_no_filter() {
        assert!(expr(vec![]).unwrap().is_none());
    }

    #[test]
    fn empty_membership_accepts_untyped_arrays() {
        let empty = || arrow::array::new_empty_array(&DataType::Null);
        for column in ["ts", "n"] {
            let none = expr(vec![vec![term(column, Op::In, empty())]]).unwrap();
            assert_eq!(none, Some(lit(false)));
            let all = expr(vec![vec![term(column, Op::NotIn, empty())]]).unwrap();
            assert_eq!(all, Some(super::column("t", column).is_not_null()));
        }
    }

    #[test]
    fn widening_and_narrowing_casts_are_lossless_only() {
        let ok = term("n", Op::Eq, Arc::new(Int64Array::from(vec![7])));
        assert!(expr(vec![vec![ok]]).is_ok());
        let overflow = term("n", Op::Eq, Arc::new(Int64Array::from(vec![i64::MAX])));
        assert!(matches!(
            expr(vec![vec![overflow]]),
            Err(Error::InvalidPredicate(_))
        ));
        let fraction = term("n", Op::Eq, Arc::new(Float64Array::from(vec![1.5])));
        assert!(matches!(
            expr(vec![vec![fraction]]),
            Err(Error::InvalidPredicate(_))
        ));
    }

    #[test]
    fn timestamps_must_be_exact_and_zoned() {
        let exact = TimestampNanosecondArray::from(vec![1_000]).with_timezone("+00:00");
        assert!(expr(vec![vec![term("ts", Op::Gt, Arc::new(exact))]]).is_ok());
        let sub_micro = TimestampNanosecondArray::from(vec![1_001]).with_timezone("UTC");
        assert!(expr(vec![vec![term("ts", Op::Gt, Arc::new(sub_micro))]]).is_err());
        let naive = TimestampNanosecondArray::from(vec![1_000]);
        assert!(expr(vec![vec![term("ts", Op::Gt, Arc::new(naive))]]).is_err());
    }

    #[test]
    fn rejects_cross_family_null_and_unknown_column() {
        let string_for_int = term("n", Op::Eq, Arc::new(StringArray::from(vec!["1"])));
        assert!(expr(vec![vec![string_for_int]]).is_err());
        let null = term("s", Op::Eq, Arc::new(StringArray::from(vec![None::<&str>])));
        assert!(expr(vec![vec![null]]).is_err());
        let unknown = term(
            "x",
            Op::IsNull,
            Arc::new(Int64Array::from(Vec::<i64>::new())),
        );
        assert!(expr(vec![vec![unknown]]).is_err());
    }

    #[test]
    fn comparison_needs_exactly_one_literal() {
        let two = term("s", Op::Eq, Arc::new(StringArray::from(vec!["a", "b"])));
        assert!(expr(vec![vec![two]]).is_err());
        let set = term("s", Op::In, Arc::new(StringArray::from(vec!["a", "b"])));
        assert!(expr(vec![vec![set]]).is_ok());
    }
}
