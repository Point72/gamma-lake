//! Features read by joining their table to the index instead of by position.
//!
//! `as_of_feature` tables hold rows at their own keys; a read takes, for every index row, the
//! table row Polars' `join_asof` would pick, matched on the non-primary sort keys.
//! `sparse_feature` tables hold only the rows that have values; a read left-joins them to the
//! index on all sort keys. Either way the result has one row per index row, so it zips with
//! the positionally aligned tables.

use std::collections::HashMap;

use arrow::array::{new_empty_array, Array, ArrayRef, AsArray, UInt32Array};
use arrow::compute::cast;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Int16Type, Int32Type, Int64Type, Int8Type,
    TimeUnit, TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt16Type, UInt32Type, UInt8Type,
};
use arrow::row::{RowConverter, Rows, SortField};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::predicate::{Dnf, Op, Term};

/// Which table row `join_asof` picks for an index row.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum Strategy {
    /// The last row at or before the key.
    #[default]
    Backward,
    /// The first row at or after the key.
    Forward,
    /// The closest row; on equal distance, the later one.
    Nearest,
}

/// Maximum distance between an index key and the matched row's key, as a magnitude.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tolerance {
    /// A number in the physical units of the key: days for dates, the time unit for
    /// timestamps and durations. Keys are integers, so a fraction is dropped.
    Units(i128),
    /// A Polars duration string such as `"1h"` or `"2d12h"`, in nanoseconds.
    Nanos(i128),
}

/// The `join_asof` parameters an `as_of_feature` stores as its `feature_params`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AsOfParams {
    pub strategy: Strategy,
    pub tolerance: Option<Tolerance>,
    pub allow_exact_matches: bool,
}

impl Default for AsOfParams {
    fn default() -> Self {
        Self {
            strategy: Strategy::Backward,
            tolerance: None,
            allow_exact_matches: true,
        }
    }
}

fn unsupported(message: String) -> Error {
    Error::UnsupportedInput(message)
}

fn invalid(message: String) -> Error {
    Error::InvalidArgument(message)
}

impl AsOfParams {
    /// Parses the `join_asof` keyword arguments stored by `add_as_of_features(params=...)`.
    ///
    /// Arguments Polars rejects are [`Error::InvalidArgument`]. Arguments that cannot change
    /// the values read are accepted and ignored; any others are [`Error::UnsupportedInput`].
    pub fn parse(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| invalid(format!("as-of feature_params are not valid JSON: {e}")))?;
        let Value::Object(map) = value else {
            return Err(invalid("as-of feature_params must be a JSON object".into()));
        };
        let mut params = AsOfParams::default();
        for (key, value) in &map {
            match (key.as_str(), value) {
                ("strategy", Value::String(s)) => {
                    params.strategy = match s.as_str() {
                        "backward" => Strategy::Backward,
                        "forward" => Strategy::Forward,
                        "nearest" => Strategy::Nearest,
                        _ => {
                            return Err(invalid(format!(
                                "as-of strategy must be backward, forward or nearest, not {s:?}"
                            )))
                        }
                    }
                }
                ("tolerance", Value::Null) => params.tolerance = None,
                ("tolerance", Value::String(s)) => {
                    params.tolerance = Some(Tolerance::Nanos(duration_nanos(s)?))
                }
                ("tolerance", Value::Number(n)) => {
                    let units = match (n.as_i64(), n.as_u64()) {
                        (Some(i), _) => i as i128,
                        (None, Some(u)) => u as i128,
                        // Saturates on overflow.
                        _ => n.as_f64().unwrap_or(0.0).trunc() as i128,
                    };
                    params.tolerance = Some(Tolerance::Units(units))
                }
                ("allow_exact_matches", Value::Bool(b)) => params.allow_exact_matches = *b,
                (
                    "check_sortedness" | "allow_parallel" | "force_parallel" | "coalesce",
                    Value::Bool(_),
                )
                | ("suffix", Value::String(_)) => {}
                _ => {
                    return Err(unsupported(format!(
                        "as-of parameter {key:?} = {value} is not supported by the Rust core"
                    )))
                }
            }
        }
        Ok(params)
    }

    /// Fails as a read with these parameters on a primary key of type `pk` would, so that
    /// unsupported reads are found while planning.
    pub fn check(&self, pk: &DataType) -> Result<()> {
        keys(new_empty_array(pk).as_ref())?;
        self.bound(pk)?;
        Ok(())
    }

    /// The tolerance for a primary key of type `pk`.
    fn bound(&self, pk: &DataType) -> Result<Option<Bound>> {
        let Some(tolerance) = &self.tolerance else {
            return Ok(None);
        };
        Ok(Some(match tolerance {
            Tolerance::Units(units) => {
                let (min, max): (i128, i128) = match pk {
                    DataType::Int8 => (i8::MIN.into(), i8::MAX.into()),
                    DataType::Int16 => (i16::MIN.into(), i16::MAX.into()),
                    DataType::Int32 | DataType::Date32 => (i32::MIN.into(), i32::MAX.into()),
                    DataType::UInt8 => (0, u8::MAX.into()),
                    DataType::UInt16 => (0, u16::MAX.into()),
                    DataType::UInt32 => (0, u32::MAX.into()),
                    _ => (i64::MIN.into(), i64::MAX.into()),
                };
                // Polars rejects some tolerances its key type cannot hold.
                if !(min..=max).contains(units) {
                    return Err(unsupported(format!(
                        "as-of tolerance {units} does not fit a {pk} primary key"
                    )));
                }
                Bound::Units(units.abs())
            }
            Tolerance::Nanos(nanos) => {
                let unit = match pk {
                    DataType::Timestamp(unit, _) | DataType::Duration(unit) => unit_nanos(unit),
                    DataType::Date32 => 86_400_000_000_000,
                    DataType::Date64 => 1_000_000,
                    _ => {
                        return Err(invalid(format!(
                            "a duration tolerance needs a temporal primary key, not {pk}"
                        )))
                    }
                };
                Bound::Nanos {
                    nanos: *nanos,
                    unit,
                }
            }
        }))
    }
}

fn unit_nanos(unit: &TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// The magnitude in nanoseconds of a Polars duration string made of fixed-length units
/// (`ns us ms s m h d w`). As in Polars, a day is always 24 hours and `""` is zero.
fn duration_nanos(s: &str) -> Result<i128> {
    let malformed = || invalid(format!("as-of tolerance {s:?} is not a valid duration"));
    let mut rest = s.strip_prefix('-').unwrap_or(s);
    let mut total = 0i128;
    while !rest.is_empty() {
        let digits = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        let unit_len = rest[digits..]
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(rest.len() - digits);
        if digits == 0 || unit_len == 0 {
            return Err(malformed());
        }
        let n: i128 = rest[..digits].parse().map_err(|_| malformed())?;
        let per: i128 = match &rest[digits..digits + unit_len] {
            "ns" => 1,
            "us" => 1_000,
            "ms" => 1_000_000,
            "s" => 1_000_000_000,
            "m" => 60_000_000_000,
            "h" => 3_600_000_000_000,
            "d" => 86_400_000_000_000,
            "w" => 604_800_000_000_000,
            "mo" | "q" | "y" => {
                return Err(invalid(format!(
                    "as-of tolerance {s:?} uses calendar units, which join_asof rejects"
                )))
            }
            "i" => {
                return Err(unsupported(format!(
                    "as-of tolerance {s:?} counts index positions"
                )))
            }
            _ => return Err(malformed()),
        };
        total = n
            .checked_mul(per)
            .and_then(|v| total.checked_add(v))
            .ok_or_else(malformed)?;
        rest = &rest[digits + unit_len..];
    }
    Ok(total)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Bound {
    Units(i128),
    Nanos { nanos: i128, unit: i128 },
}

/// Whether two keys `distance` apart are within `bound`.
fn within(distance: i128, bound: Bound) -> bool {
    match bound {
        Bound::Units(units) => distance <= units,
        Bound::Nanos { nanos, unit } => distance.saturating_mul(unit) <= nanos,
    }
}

/// Primary key values as integers in the key's physical units. Floating-point keys are
/// [`Error::UnsupportedInput`]: Polars orders and measures them in ways (NaN, `f32`
/// arithmetic) that are not reproduced here.
fn keys(array: &dyn Array) -> Result<Vec<i64>> {
    macro_rules! ints {
        ($t:ty) => {
            array
                .as_primitive::<$t>()
                .values()
                .iter()
                .map(|&v| v as i64)
                .collect()
        };
    }
    Ok(match array.data_type() {
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt32 => ints!(UInt32Type),
        DataType::Date32 => ints!(Date32Type),
        DataType::Date64 => ints!(Date64Type),
        DataType::Timestamp(TimeUnit::Second, _) => ints!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => ints!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => ints!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => ints!(TimestampNanosecondType),
        DataType::Duration(TimeUnit::Second) => ints!(DurationSecondType),
        DataType::Duration(TimeUnit::Millisecond) => ints!(DurationMillisecondType),
        DataType::Duration(TimeUnit::Microsecond) => ints!(DurationMicrosecondType),
        DataType::Duration(TimeUnit::Nanosecond) => ints!(DurationNanosecondType),
        other => {
            return Err(unsupported(format!(
                "as-of features need an integer or temporal primary sort key, not {other}"
            )))
        }
    })
}

/// The position in `sorted` of the row `join_asof` matches with key `t`, or `None`.
fn pick(sorted: &[i64], t: i64, params: &AsOfParams, bound: Option<Bound>) -> Option<usize> {
    let exact = params.allow_exact_matches;
    let before = || {
        let n = sorted.partition_point(|&v| if exact { v <= t } else { v < t });
        n.checked_sub(1)
    };
    let after = || {
        let n = sorted.partition_point(|&v| if exact { v < t } else { v <= t });
        (n < sorted.len()).then_some(n)
    };
    let gap = |i: usize| (t as i128 - sorted[i] as i128).abs();
    let chosen = match params.strategy {
        Strategy::Backward => before(),
        Strategy::Forward => after(),
        Strategy::Nearest => {
            // Ties go to the later row, including the last of several rows with the same key.
            let after = after().map(|i| sorted.partition_point(|&w| w <= sorted[i]) - 1);
            match (before(), after) {
                (Some(b), Some(a)) => Some(if gap(a) <= gap(b) { a } else { b }),
                (b, a) => b.or(a),
            }
        }
    }?;
    match bound {
        Some(bound) if !within(gap(chosen), bound) => None,
        _ => Some(chosen),
    }
}

/// Whether row `i` of any of `columns` is null. Keys with a null never match, as in Polars.
fn any_null(columns: &[ArrayRef], i: usize) -> bool {
    columns.iter().any(|c| c.is_null(i))
}

fn converter(columns: &[ArrayRef]) -> Result<Option<RowConverter>> {
    if columns.is_empty() {
        return Ok(None);
    }
    let fields = columns
        .iter()
        .map(|c| SortField::new(c.data_type().clone()))
        .collect();
    Ok(Some(RowConverter::new(fields)?))
}

/// `columns` cast to the types `converter` was built for, as rows.
fn rows(converter: &RowConverter, columns: &[ArrayRef], types: &[DataType]) -> Result<Rows> {
    let columns = columns
        .iter()
        .zip(types)
        .map(|(c, t)| {
            if c.data_type() == t {
                Ok(c.clone())
            } else {
                cast(c, t)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(converter.convert_columns(&columns)?)
}

/// Primary keys of a group in order, and the row each came from.
type Group = (Vec<i64>, Vec<u32>);

/// An `as_of_feature` table prepared for lookups: row numbers grouped by the non-primary sort
/// keys, each group ordered by the primary key.
pub(crate) struct AsOfTable {
    params: AsOfParams,
    bound: Option<Bound>,
    pk_type: DataType,
    by_types: Vec<DataType>,
    converter: Option<RowConverter>,
    groups: HashMap<Box<[u8]>, Group>,
}

impl AsOfTable {
    /// Prepares the table with primary key column `pk` and other sort key columns `by`.
    /// Rows with a null key never match.
    pub fn new(params: &AsOfParams, by: &[ArrayRef], pk: &dyn Array) -> Result<Self> {
        let pk_type = pk.data_type().clone();
        let bound = params.bound(&pk_type)?;
        let values = keys(pk)?;
        let converter = converter(by)?;
        let by_types: Vec<DataType> = by.iter().map(|c| c.data_type().clone()).collect();
        let by_rows = converter
            .as_ref()
            .map(|c| rows(c, by, &by_types))
            .transpose()?;
        let mut members: HashMap<Box<[u8]>, Vec<u32>> = HashMap::new();
        for i in (0..pk.len()).filter(|&i| pk.is_valid(i) && !any_null(by, i)) {
            let key: &[u8] = by_rows.as_ref().map_or(&[], |r| r.row(i).data());
            match members.get_mut(key) {
                Some(rows) => rows.push(i as u32),
                None => {
                    members.insert(key.into(), vec![i as u32]);
                }
            }
        }
        let groups = members
            .into_iter()
            .map(|(key, mut rows)| {
                rows.sort_by_key(|&r| values[r as usize]);
                let sorted = rows.iter().map(|&r| values[r as usize]).collect();
                (key, (sorted, rows))
            })
            .collect();
        Ok(Self {
            params: params.clone(),
            bound,
            pk_type,
            by_types,
            converter,
            groups,
        })
    }

    /// For each index row, the table row it matches, or null.
    pub fn lookup(&self, by: &[ArrayRef], pk: &ArrayRef) -> Result<UInt32Array> {
        let pk = if pk.data_type() == &self.pk_type {
            pk.clone()
        } else {
            cast(pk, &self.pk_type)?
        };
        let values = keys(pk.as_ref())?;
        let by_rows = self
            .converter
            .as_ref()
            .map(|c| rows(c, by, &self.by_types))
            .transpose()?;
        Ok((0..pk.len())
            .map(|i| {
                if pk.is_null(i) || any_null(by, i) {
                    return None;
                }
                let key: &[u8] = by_rows.as_ref().map_or(&[], |r| r.row(i).data());
                let (sorted, rows) = self.groups.get(key)?;
                pick(sorted, values[i], &self.params, self.bound).map(|j| rows[j])
            })
            .collect())
    }
}

/// A `sparse_feature` table prepared for lookups by all sort keys.
pub(crate) struct SparseTable {
    types: Vec<DataType>,
    converter: RowConverter,
    rows: HashMap<Box<[u8]>, u32>,
}

impl SparseTable {
    /// Prepares the table with sort key columns `keys`; `table_addr` names it in errors.
    pub fn new(keys: &[ArrayRef], table_addr: &str) -> Result<Self> {
        let converter = converter(keys)?
            .ok_or_else(|| Error::InvalidArgument("a lake needs at least one sort key".into()))?;
        let types: Vec<DataType> = keys.iter().map(|c| c.data_type().clone()).collect();
        let converted = rows(&converter, keys, &types)?;
        let mut rows = HashMap::with_capacity(converted.num_rows());
        for (i, row) in converted.iter().enumerate() {
            if any_null(keys, i) {
                continue;
            }
            if rows.insert(Box::from(row.data()), i as u32).is_some() {
                return Err(Error::Integrity {
                    table_addr: table_addr.to_string(),
                    kind: "a sparse feature table holds the same sort keys twice".into(),
                });
            }
        }
        Ok(Self {
            types,
            converter,
            rows,
        })
    }

    /// For each index row, the table row with the same keys, or null.
    pub fn lookup(&self, keys: &[ArrayRef]) -> Result<UInt32Array> {
        let converted = rows(&self.converter, keys, &self.types)?;
        Ok(converted
            .iter()
            .enumerate()
            .map(|(i, row)| {
                if any_null(keys, i) {
                    return None;
                }
                self.rows.get(row.data()).copied()
            })
            .collect())
    }
}

/// Most values of an `In` on the primary key that are rewritten into bounds; longer lists
/// are dropped from the table predicate instead.
const MAX_BOUNDED_VALUES: usize = 16;

/// A predicate on an as-of table that keeps every row a read of the index filtered by `dnf`
/// can match. `pk` is the primary sort key.
///
/// Terms on other sort keys are kept, since rows only match within equal keys. For
/// `Backward`, an index row at or below a bound only matches table rows at or below it, so
/// `<` and `<=` are kept, `=` becomes `<=`, and `In` becomes `<=` any of its values; other
/// terms on `pk` are dropped. `Forward` is the mirror image. `Nearest` can match in either
/// direction, so it drops every term on `pk`.
pub(crate) fn table_predicate(dnf: &Dnf, pk: &str, strategy: Strategy) -> Dnf {
    let bound = match strategy {
        Strategy::Backward => Some(Op::LtEq),
        Strategy::Forward => Some(Op::GtEq),
        Strategy::Nearest => None,
    };
    let mut clauses: Vec<Vec<Term>> = vec![];
    for clause in &dnf.0 {
        let mut kept: Vec<Term> = vec![];
        let mut alternatives: Option<Vec<Term>> = None;
        for term in clause {
            if term.column != pk {
                kept.push(term.clone());
                continue;
            }
            let Some(bound) = bound else { continue };
            match (strategy, term.op) {
                (Strategy::Backward, Op::Lt | Op::LtEq)
                | (Strategy::Forward, Op::Gt | Op::GtEq) => kept.push(term.clone()),
                (_, Op::Eq) => kept.push(Term {
                    op: bound,
                    ..term.clone()
                }),
                (_, Op::In)
                    if alternatives.is_none() && term.values.len() <= MAX_BOUNDED_VALUES =>
                {
                    alternatives = Some(
                        (0..term.values.len())
                            .map(|i| Term {
                                column: term.column.clone(),
                                op: bound,
                                values: term.values.slice(i, 1),
                            })
                            .collect(),
                    )
                }
                _ => {}
            }
        }
        match alternatives {
            // `In` with no values matches nothing, and neither does its clause.
            Some(alternatives) => clauses.extend(alternatives.into_iter().map(|alternative| {
                let mut clause = kept.clone();
                clause.push(alternative);
                clause
            })),
            None => clauses.push(kept),
        }
    }
    if clauses.iter().any(Vec::is_empty) {
        Dnf::default()
    } else {
        Dnf(clauses)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};

    use super::*;

    fn params(json: &str) -> AsOfParams {
        AsOfParams::parse(json).unwrap()
    }

    fn matches(json: &str, right: Vec<i64>, left: Vec<i64>) -> Vec<Option<u32>> {
        let table = AsOfTable::new(&params(json), &[], &Int64Array::from(right)).unwrap();
        let left: ArrayRef = Arc::new(Int64Array::from(left));
        table.lookup(&[], &left).unwrap().iter().collect()
    }

    #[test]
    fn strategies_match_polars() {
        // Cases probed against polars 1.44 `join_asof`; right keys [3, 3, 7, 7, 10].
        let right = vec![3, 3, 7, 7, 10];
        let left = vec![5, 6, 10];
        let cases = [
            (r#"{}"#, [Some(1), Some(1), Some(4)]),
            (
                r#"{"allow_exact_matches": false}"#,
                [Some(1), Some(1), Some(3)],
            ),
            (r#"{"strategy": "forward"}"#, [Some(2), Some(2), Some(4)]),
            (
                r#"{"strategy": "forward", "allow_exact_matches": false}"#,
                [Some(2), Some(2), None],
            ),
            (r#"{"strategy": "nearest"}"#, [Some(3), Some(3), Some(4)]),
            (
                r#"{"strategy": "nearest", "allow_exact_matches": false}"#,
                [Some(3), Some(3), Some(3)],
            ),
            (
                r#"{"strategy": "nearest", "tolerance": 1}"#,
                [None, Some(3), Some(4)],
            ),
        ];
        for (json, expected) in cases {
            assert_eq!(
                matches(json, right.clone(), left.clone()),
                expected,
                "{json}"
            );
        }
        // A tie in distance goes to the later row.
        assert_eq!(
            matches(r#"{"strategy": "nearest"}"#, vec![4, 6], vec![5]),
            [Some(1)]
        );
    }

    #[test]
    fn tolerance_is_inclusive_and_groups_are_separate() {
        let by: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["a", "b", "a"]))];
        let pk = TimestampMicrosecondArray::from(vec![0, 0, 3_600_000_000]).with_timezone("UTC");
        let table = AsOfTable::new(&params(r#"{"tolerance": "1h"}"#), &by, &pk).unwrap();
        let left_by: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["a", "b", "b", "c"]))];
        let left_pk: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![7_200_000_000, 3_600_000_000, 3_600_000_001, 0])
                .with_timezone("UTC"),
        );
        let found: Vec<_> = table.lookup(&left_by, &left_pk).unwrap().iter().collect();
        assert_eq!(found, [Some(2), Some(1), None, None]);

        // A null by-key matches nothing, not even a null by-key.
        let by: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![None, Some("a")]))];
        let pk = Int64Array::from(vec![0, 0]);
        let table = AsOfTable::new(&AsOfParams::default(), &by, &pk).unwrap();
        let left_pk: ArrayRef = Arc::new(Int64Array::from(vec![1, 1]));
        let found: Vec<_> = table.lookup(&by, &left_pk).unwrap().iter().collect();
        assert_eq!(found, [None, Some(1)]);
    }

    #[test]
    fn parameters_polars_rejects_are_invalid() {
        for json in [
            "[]",
            r#"{"strategy": "sideways"}"#,
            r#"{"tolerance": "1mo"}"#,
            r#"{"tolerance": "2y"}"#,
            r#"{"tolerance": "h"}"#,
            r#"{"tolerance": " 1h"}"#,
            r#"{"tolerance": "1.5d"}"#,
            r#"{"tolerance": "1h-1s"}"#,
        ] {
            assert!(
                matches!(AsOfParams::parse(json), Err(Error::InvalidArgument(_))),
                "{json}"
            );
        }
        for json in [
            r#"{"by": "x"}"#,
            r#"{"tolerance": "3i"}"#,
            r#"{"suffix": 1}"#,
        ] {
            assert!(
                matches!(AsOfParams::parse(json), Err(Error::UnsupportedInput(_))),
                "{json}"
            );
        }
        let duration = params(r#"{"tolerance": "1h"}"#);
        assert!(matches!(
            duration.bound(&DataType::Int64),
            Err(Error::InvalidArgument(_))
        ));
        assert!(AsOfTable::new(&duration, &[], &StringArray::from(vec!["x"])).is_err());
        let backward = AsOfParams::default();
        assert!(backward.check(&DataType::Int64).is_ok());
        for pk in [DataType::Float64, DataType::Utf8, DataType::UInt64] {
            assert!(
                matches!(backward.check(&pk), Err(Error::UnsupportedInput(_))),
                "{pk}"
            );
        }
        assert!(matches!(
            duration.check(&DataType::Int64),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn tolerances_follow_polars() {
        // Probed against polars 1.44: numbers are in the key's physical units, by magnitude;
        // durations compare in nanoseconds, with a day always 24 hours.
        let ignored = params(r#"{"suffix": "_x", "coalesce": false, "check_sortedness": true}"#);
        assert_eq!(ignored, AsOfParams::default());
        assert_eq!(
            params(r#"{"tolerance": "-1d2h"}"#).tolerance,
            Some(Tolerance::Nanos(93_600_000_000_000))
        );
        assert_eq!(
            params(r#"{"tolerance": ""}"#).tolerance,
            Some(Tolerance::Nanos(0))
        );
        let within = |json: &str, pk: DataType, distance: i64| {
            let bound = params(json).bound(&pk).unwrap().unwrap();
            within(distance as i128, bound)
        };
        let us = DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into()));
        assert!(within(r#"{"tolerance": 1000000}"#, us.clone(), 1_000_000));
        assert!(!within(r#"{"tolerance": 1000000}"#, us.clone(), 1_000_001));
        assert!(within(r#"{"tolerance": "1d"}"#, us.clone(), 86_400_000_000));
        assert!(!within(r#"{"tolerance": "1d"}"#, us, 90_000_000_000));
        assert!(within(r#"{"tolerance": 1}"#, DataType::Date32, 1));
        assert!(within(r#"{"tolerance": "25h"}"#, DataType::Date32, 1));
        assert!(!within(r#"{"tolerance": "23h"}"#, DataType::Date32, 1));
        assert!(within(r#"{"tolerance": 2.5}"#, DataType::Int64, 2));
        assert!(!within(r#"{"tolerance": 1.9}"#, DataType::Int64, 2));
        assert!(within(r#"{"tolerance": -1}"#, DataType::Int64, 1));
        // Integer tolerances are exact beyond 2^53.
        assert!(!within(
            r#"{"tolerance": 9007199254740992}"#,
            DataType::Int64,
            9_007_199_254_740_993
        ));
        let ms = DataType::Timestamp(TimeUnit::Millisecond, None);
        assert!(within(r#"{"tolerance": "1001us"}"#, ms.clone(), 1));
        assert!(!within(r#"{"tolerance": "999us"}"#, ms, 1));
        assert!(within(
            r#"{"tolerance": -9223372036854775808}"#,
            DataType::Int64,
            1
        ));
        assert!(within(r#"{"tolerance": -2147483648}"#, DataType::Int32, 1));
        // Polars rejects tolerances outside the key's physical type.
        for (json, pk) in [
            (r#"{"tolerance": 9223372036854775808}"#, DataType::Int64),
            (r#"{"tolerance": 1e40}"#, DataType::Int64),
            (r#"{"tolerance": 2147483648}"#, DataType::Int32),
            (r#"{"tolerance": -1}"#, DataType::UInt32),
        ] {
            assert!(matches!(
                params(json).check(&pk),
                Err(Error::UnsupportedInput(_))
            ));
        }
    }

    #[test]
    fn sparse_lookups_miss_with_null_and_reject_duplicates() {
        let keys: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "a"])),
        ];
        let table = SparseTable::new(&keys, "t").unwrap();
        let left: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![2, 1, 1])),
            Arc::new(StringArray::from(vec!["a", "b", "a"])),
        ];
        let found: Vec<_> = table.lookup(&left).unwrap().iter().collect();
        assert_eq!(found, [Some(1), None, Some(0)]);
        // Null keys match nothing, not even each other, and are not duplicates.
        let nulls: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![None, None, Some(1)]))];
        let table = SparseTable::new(&nulls, "t").unwrap();
        let found: Vec<_> = table.lookup(&nulls).unwrap().iter().collect();
        assert_eq!(found, [None, None, Some(2)]);
        let dup: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![1, 1]))];
        assert!(matches!(
            SparseTable::new(&dup, "t"),
            Err(Error::Integrity { .. })
        ));
    }

    #[test]
    fn table_predicates_keep_every_row_the_index_can_match() {
        let term = |column: &str, op, values: Vec<i64>| Term {
            column: column.into(),
            op,
            values: Arc::new(Int64Array::from(values)),
        };
        let show = |dnf: &Dnf| {
            dnf.0
                .iter()
                .map(|clause| {
                    clause
                        .iter()
                        .map(|t| {
                            let values = t.values.as_primitive::<Int64Type>().values().to_vec();
                            format!("{}{:?}{values:?}", t.column, t.op)
                        })
                        .collect::<Vec<_>>()
                        .join("&")
                })
                .collect::<Vec<_>>()
        };
        let window = Dnf(vec![vec![
            term("t", Op::GtEq, vec![3]),
            term("t", Op::Lt, vec![5]),
            term("s", Op::Eq, vec![1]),
        ]]);
        assert_eq!(
            show(&table_predicate(&window, "t", Strategy::Backward)),
            ["tLt[5]&sEq[1]"]
        );
        assert_eq!(
            show(&table_predicate(&window, "t", Strategy::Forward)),
            ["tGtEq[3]&sEq[1]"]
        );
        assert_eq!(
            show(&table_predicate(&window, "t", Strategy::Nearest)),
            ["sEq[1]"]
        );
        let points = Dnf(vec![
            vec![term("t", Op::In, vec![2, 9]), term("t", Op::NotEq, vec![4])],
            vec![term("t", Op::Eq, vec![7])],
        ]);
        assert_eq!(
            show(&table_predicate(&points, "t", Strategy::Backward)),
            ["tLtEq[2]", "tLtEq[9]", "tLtEq[7]"]
        );
        // A clause left without terms matches everything; one with an empty In nothing.
        let open = Dnf(vec![
            vec![term("t", Op::Gt, vec![3])],
            vec![term("s", Op::Eq, vec![1])],
        ]);
        assert!(table_predicate(&open, "t", Strategy::Backward).is_empty());
        let empty = Dnf(vec![
            vec![term("t", Op::In, vec![])],
            vec![term("s", Op::Eq, vec![1])],
        ]);
        assert_eq!(
            show(&table_predicate(&empty, "t", Strategy::Forward)),
            ["sEq[1]"]
        );
        let long = Dnf(vec![vec![term("t", Op::In, (0..17).collect())]]);
        assert!(table_predicate(&long, "t", Strategy::Backward).is_empty());
    }
}
