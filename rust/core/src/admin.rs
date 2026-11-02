//! Annotations, restore and consolidation.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, PoisonError};

use arrow::array::{
    Array, ArrayRef, AsArray, Int64Array, ListBuilder, RecordBatch, StringArray, StringBuilder,
    TimestampMicrosecondArray, UInt32Array,
};
use arrow::compute::{cast, take_record_batch};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef};
use arrow::record_batch::RecordBatchReader;
use arrow::row::{RowConverter, SortField};

use crate::error::{Error, Result};
use crate::lake::{feature_rows, latest, FeatureRow, Lake, SelectedFeature};
use crate::predicate::{Dnf, Op, Term};
use crate::runtime::block_on;
use crate::store::StoredTable;
use crate::write::{
    assemble, cast_into, commit, coverage_values, integrity, new_table_addr, now_micros,
    write_lock, Action, Commits,
};

/// A column of a table and the metadata keys to set on it.
type ColumnUpdate = (String, HashMap<String, String>);

const COMMENT: &str = "comment";
const TAGS: &str = "tags";
const CONSOLIDATABLE_SIGNAL_TYPES: [&str; 2] = ["feature", "target"];

/// A comment and tags for one version of a feature.
///
/// They are stored as the `comment` and `tags` (a JSON list of strings) metadata of the
/// feature's column in its table, as `GammaFeatureLake.annotate` stores them, so every
/// version stored in that table shares them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Annotation {
    pub feature_name: String,
    /// The table holding the annotated version; `None` for the latest version.
    pub table_addr: Option<String>,
    /// Replaces the stored comment; `None` keeps it.
    pub comment: Option<String>,
    /// Replace the stored tags; `None` keeps them.
    pub tags: Option<Vec<String>>,
}

/// Columns of [`Lake::annotations`].
fn annotations_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("feature_name", DataType::Utf8, false),
        Field::new("table_addr", DataType::Utf8, false),
        Field::new(COMMENT, DataType::Utf8, true),
        tags_field(),
    ]))
}

fn tags_field() -> Field {
    Field::new(
        TAGS,
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        false,
    )
}

/// Whether `row` describes a feature stored in a table: not runtime-computed, with an address.
fn stored(row: &FeatureRow) -> Option<&str> {
    match (&row.table_addr, row.signal_type.as_deref()) {
        (_, Some("runtime_computed")) => None,
        (Some(addr), _) if addr != "n/a" => Some(addr),
        _ => None,
    }
}

/// `error` as a [`Error::Write`] when tables were already committed.
pub(crate) fn after(phase: String, committed: Vec<String>, error: Error) -> Error {
    match error {
        Error::Write {
            committed: more,
            phase,
            source,
        } => Error::Write {
            phase,
            committed: committed.into_iter().chain(more).collect(),
            source,
        },
        error if committed.is_empty() => error,
        error => Error::Write {
            phase,
            committed,
            source: Box::new(error),
        },
    }
}

impl Lake {
    /// Sets comments and tags of feature versions, as `GammaFeatureLake.annotate` does.
    ///
    /// Annotations without a comment or tags are skipped. Every annotation is checked before
    /// the first one is stored; a later failure is an [`Error::Write`]. Each annotated column
    /// is one commit to its table, and annotations of one column are applied in order.
    pub fn annotate(&self, annotations: &[Annotation]) -> Result<()> {
        let lock = write_lock(self.store.location());
        let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        block_on(self.annotate_locked(annotations, vec![], &Commits::default()))?
    }

    /// [`Lake::annotate`] with the write lock held, after `committed` tables were written,
    /// counting its commits in `commits`.
    pub(crate) async fn annotate_locked(
        &self,
        annotations: &[Annotation],
        committed: Vec<String>,
        commits: &Commits,
    ) -> Result<()> {
        let phase = "annotating features".to_string();
        let annotations: Vec<&Annotation> = annotations
            .iter()
            .filter(|a| a.comment.is_some() || a.tags.is_some())
            .collect();
        if annotations.is_empty() {
            return Ok(());
        }
        let plan = async {
            let fm = self.read_batch(&self.options.feature_metadata_path).await?;
            let rows = feature_rows(&fm)?;
            let best = latest(&rows);
            let mut missing = vec![];
            // Updates per table and column, in first-appearance order.
            let mut tables: Vec<(String, Vec<ColumnUpdate>)> = vec![];
            for a in &annotations {
                let name = &a.feature_name;
                let addr = match &a.table_addr {
                    Some(addr) => {
                        if !rows
                            .iter()
                            .any(|r| &r.name == name && r.table_addr.as_ref() == Some(addr))
                        {
                            missing.push(format!("{name} in table {addr}"));
                            continue;
                        }
                        addr.clone()
                    }
                    None => match best.get(name).map(|&i| &rows[i]) {
                        None => {
                            missing.push(name.clone());
                            continue;
                        }
                        Some(row) => stored(row)
                            .ok_or_else(|| {
                                Error::InvalidArgument(format!(
                                    "feature {name:?} is not stored in a table, so it cannot \
                                     be annotated"
                                ))
                            })?
                            .to_string(),
                    },
                };
                let mut update = HashMap::new();
                if let Some(comment) = &a.comment {
                    update.insert(COMMENT.to_string(), comment.clone());
                }
                if let Some(tags) = &a.tags {
                    update.insert(TAGS.to_string(), serde_json::to_string(tags).unwrap());
                }
                let i = match tables.iter().position(|(t, _)| *t == addr) {
                    Some(i) => i,
                    None => {
                        tables.push((addr, vec![]));
                        tables.len() - 1
                    }
                };
                let columns = &mut tables[i].1;
                match columns.iter_mut().find(|(c, _)| c == name) {
                    Some((_, metadata)) => metadata.extend(update),
                    None => columns.push((name.clone(), update)),
                }
            }
            if !missing.is_empty() {
                return Err(Error::MissingFeatures(missing));
            }
            let mut opened = vec![];
            for (addr, columns) in tables {
                let table = self.table(&addr).await?;
                let schema = table.schema();
                if let Some((column, _)) = columns
                    .iter()
                    .find(|(c, _)| schema.field_with_name(c).is_err())
                {
                    return Err(Error::InvalidArgument(format!(
                        "table {addr} has no column {column:?}"
                    )));
                }
                opened.push((addr, table, columns));
            }
            Ok(opened)
        };
        let opened = plan
            .await
            .map_err(|e| after(phase.clone(), committed.clone(), e))?;
        let mut committed = committed;
        // table version (`next`), which the following commit must build on. Every
        // commit is tagged with the write's transaction id.
        for (addr, table, columns) in opened {
            let mut table = commits.tagged(table);
            for (column, metadata) in columns {
                let read = table.version();
                let result = table.set_column_metadata(&column, metadata).await;
                match commits.count(&addr, read, result) {
                    Ok(next) => table = next,
                    Err(e) => {
                        return Err(after(format!("annotating table {addr}"), committed, e));
                    }
                }
                if committed.last() != Some(&addr) {
                    committed.push(addr.clone());
                }
            }
        }
        Ok(())
    }

    /// The comment and tags of each `(feature_name, table_addr)`, one row per pair in order,
    /// with columns `feature_name`, `table_addr`, `comment` and `tags`. Unannotated features
    /// have a null comment and no tags.
    pub fn annotations(&self, features: &[(&str, &str)]) -> Result<RecordBatch> {
        block_on(self.annotations_of(features))?
    }

    async fn annotations_of(&self, features: &[(&str, &str)]) -> Result<RecordBatch> {
        let mut schemas: HashMap<&str, SchemaRef> = HashMap::new();
        for (_, addr) in features {
            if !schemas.contains_key(addr) {
                schemas.insert(addr, self.table(addr).await?.schema());
            }
        }
        let mut comments = StringBuilder::new();
        let mut tags = ListBuilder::new(StringBuilder::new())
            .with_field(Arc::new(Field::new_list_field(DataType::Utf8, true)));
        for (name, addr) in features {
            let metadata = schemas[addr]
                .field_with_name(name)
                .map(|f| f.metadata().clone())
                .unwrap_or_default();
            comments.append_option(metadata.get(COMMENT));
            match metadata.get(TAGS) {
                None => {}
                Some(json) => {
                    let values: Vec<String> = serde_json::from_str(json).map_err(|e| {
                        integrity(
                            addr,
                            format!("tags of column {name:?} are not a JSON list of strings: {e}"),
                        )
                    })?;
                    for v in values {
                        tags.values().append_value(v);
                    }
                }
            }
            tags.append(true);
        }
        Ok(RecordBatch::try_new(
            annotations_schema(),
            vec![
                Arc::new(StringArray::from_iter_values(features.iter().map(|f| f.0))),
                Arc::new(StringArray::from_iter_values(features.iter().map(|f| f.1))),
                Arc::new(comments.finish()),
                Arc::new(tags.finish()),
            ],
        )?)
    }

    /// The latest feature metadata rows of `features`, in order, or of every feature when
    /// empty, with the `comment` and `tags` of each version. Runtime-computed features have
    /// a null comment and no tags.
    pub fn describe(&self, features: &[&str]) -> Result<RecordBatch> {
        block_on(async {
            let fm = self.read_batch(&self.options.feature_metadata_path).await?;
            let rows = feature_rows(&fm)?;
            let best = latest(&rows);
            let chosen: Vec<&FeatureRow> = if features.is_empty() {
                let mut all: Vec<usize> = best.into_values().collect();
                all.sort_unstable_by_key(|&i| rows[i].row);
                all.into_iter().map(|i| &rows[i]).collect()
            } else {
                let mut seen = HashSet::new();
                let wanted: Vec<&str> = features
                    .iter()
                    .copied()
                    .filter(|n| seen.insert(*n))
                    .collect();
                let missing: Vec<String> = wanted
                    .iter()
                    .filter(|n| !best.contains_key(**n))
                    .map(|n| n.to_string())
                    .collect();
                if !missing.is_empty() {
                    return Err(Error::MissingFeatures(missing));
                }
                wanted.iter().map(|n| &rows[best[*n]]).collect()
            };
            let pairs: Vec<(&str, &str)> = chosen
                .iter()
                .filter_map(|r| stored(r).map(|a| (r.name.as_str(), a)))
                .collect();
            let found = self.annotations_of(&pairs).await?;
            // Rows of `found` per chosen row; runtime-computed rows take an empty annotation.
            let empty = RecordBatch::try_new(
                annotations_schema(),
                vec![
                    Arc::new(StringArray::from(vec![""])),
                    Arc::new(StringArray::from(vec![""])),
                    Arc::new(StringArray::from(vec![None::<&str>])),
                    Arc::new({
                        let mut b = ListBuilder::new(StringBuilder::new())
                            .with_field(Arc::new(Field::new_list_field(DataType::Utf8, true)));
                        b.append(true);
                        b.finish()
                    }),
                ],
            )?;
            let both = arrow::compute::concat_batches(&annotations_schema(), [&found, &empty])?;
            let mut next = 0u32;
            let picks: Vec<u32> = chosen
                .iter()
                .map(|r| match stored(r) {
                    Some(_) => {
                        next += 1;
                        next - 1
                    }
                    None => found.num_rows() as u32,
                })
                .collect();
            let picks = take_record_batch(&both, &UInt32Array::from(picks))?;
            let metadata = take_record_batch(
                &fm,
                &UInt32Array::from_iter_values(chosen.iter().map(|r| r.row as u32)),
            )?;
            let mut fields: Vec<Field> = metadata
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            fields.push(Field::new(COMMENT, DataType::Utf8, true));
            fields.push(tags_field());
            let mut columns = metadata.columns().to_vec();
            columns.push(picks.column(2).clone());
            columns.push(picks.column(3).clone());
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                columns,
            )?)
        })?
    }

    /// Restores the lake to how it was at `at`, in microseconds since the Unix epoch, as
    /// `GammaFeatureLake.restore_to_timestamp` does.
    ///
    /// The index and the metadata tables are restored first, then every table the restored
    /// feature metadata refers to. A table with no commit after `at` is left as it is.
    /// Tables created after `at` are left in place, unreferenced. Restore is not atomic: a
    /// failure after the first table was restored is an [`Error::Write`].
    pub fn restore(&self, at: i64) -> Result<()> {
        let lock = write_lock(self.store.location());
        let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        block_on(async {
            let mut committed = vec![];
            let restore = |addr: String, committed: Vec<String>| async move {
                let phase = format!("restoring table {addr}");
                let result = async { self.table(&addr).await?.restore(at).await }.await;
                match result {
                    Ok(true) => Ok((committed.into_iter().chain([addr]).collect(), ())),
                    Ok(false) => Ok((committed, ())),
                    Err(e) => Err(after(phase, committed, e)),
                }
            };
            for addr in [
                &self.options.index_path,
                &self.options.table_metadata_path,
                &self.options.feature_metadata_path,
            ] {
                (committed, ()) = restore(addr.clone(), committed).await?;
            }
            let fm = match self.read_batch(&self.options.feature_metadata_path).await {
                Ok(fm) => fm,
                Err(e) => return Err(after("reading feature metadata".into(), committed, e)),
            };
            let rows = match feature_rows(&fm) {
                Ok(rows) => rows,
                Err(e) => return Err(after("reading feature metadata".into(), committed, e)),
            };
            let addrs: BTreeSet<&str> = rows.iter().filter_map(stored).collect();
            for addr in addrs {
                (committed, ()) = restore(addr.to_string(), committed).await?;
            }
            Ok(())
        })?
    }

    /// Moves the latest versions of `features` into one new table, as
    /// `GammaFeatureLake.consolidate_feature_groups` does, and returns its address, or
    /// `None` when they already share a table.
    ///
    /// Only `feature` and `target` signals can be consolidated. The new table holds the
    /// index rows up to the newest coverage of the tables merged, and each feature keeps
    /// its comment and tags. Each feature gets a new version that differs only in its
    /// table; reads return the same values.
    pub fn consolidate(&self, features: &[&str]) -> Result<Option<String>> {
        let lock = write_lock(self.store.location());
        let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        let plan = block_on(self.consolidation(features))??;
        let Some(plan) = plan else {
            return Ok(None);
        };
        let pk = &self.options.primary_sort_key;
        let dnf = Dnf(vec![vec![Term {
            column: pk.clone(),
            op: Op::LtEq,
            values: plan.coverage.clone(),
        }]]);
        let read = self.read_selected(&plan.selected, &dnf)?;
        let schema = read.schema();
        let batches = read.collect::<std::result::Result<Vec<_>, _>>()?;
        let read = arrow::compute::concat_batches(&schema, &batches)?;
        let addr = new_table_addr(&self.options.table_name_prefix);
        let columns = plan
            .schema
            .fields()
            .iter()
            .map(|f| {
                let values = read
                    .column_by_name(f.name())
                    .ok_or_else(|| integrity(&addr, format!("read lacks column {:?}", f.name())))?;
                cast_into(values, f, &addr)
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = RecordBatch::try_new(plan.schema.clone(), columns)?;
        let Consolidation { tm, fm, .. } = plan;
        block_on(async {
            let tm_batch = assemble(
                &tm.schema(),
                HashMap::from([
                    (
                        "table_addr",
                        Arc::new(StringArray::from(vec![addr.as_str()])) as ArrayRef,
                    ),
                    ("last_updated", plan.coverage.clone()),
                    (
                        "update_timestamp",
                        Arc::new(
                            TimestampMicrosecondArray::from(vec![now_micros()])
                                .with_timezone("UTC"),
                        ),
                    ),
                ]),
                1,
                &self.options.table_metadata_path,
            )?;
            let fm_schema = fm.schema();
            let n = plan.base.num_rows();
            let versions = cast(
                plan.base
                    .column_by_name("version")
                    .ok_or_else(|| integrity("feature_metadata", "missing column \"version\""))?,
                &DataType::Int64,
            )?;
            let versions: Int64Array = versions
                .as_primitive::<Int64Type>()
                .iter()
                .map(|v| v.map(|v| v + 1))
                .collect();
            let mut columns: HashMap<&str, ArrayRef> = HashMap::new();
            for field in fm_schema.fields() {
                if let Some(values) = plan.base.column_by_name(field.name()) {
                    columns.insert(field.name().as_str(), values.clone());
                }
            }
            columns.insert("version", Arc::new(versions));
            columns.insert(
                "table_addr",
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    addr.as_str(),
                    n,
                ))),
            );
            let fm_batch = assemble(&fm_schema, columns, n, &self.options.feature_metadata_path)?;
            commit(
                self,
                vec![Action::Create {
                    addr: addr.clone(),
                    batch,
                }],
                None,
                vec![
                    Action::Append {
                        addr: self.options.table_metadata_path.clone(),
                        table: tm,
                        batch: tm_batch,
                    },
                    Action::Append {
                        addr: self.options.feature_metadata_path.clone(),
                        table: fm,
                        batch: fm_batch,
                    },
                ],
                &Commits::default(),
            )
            .await
        })??;
        Ok(Some(addr))
    }

    async fn consolidation(&self, features: &[&str]) -> Result<Option<Consolidation>> {
        // The metadata tables are committed to through the handles read here, so the commit
        // fails if another writer changed them in between.
        let fm_table = self.table(&self.options.feature_metadata_path).await?;
        let tm_table = self.table(&self.options.table_metadata_path).await?;
        let fm = self.read_all(fm_table.as_ref()).await?;
        let tm = self.read_all(tm_table.as_ref()).await?;
        let rows = feature_rows(&fm)?;
        let best = latest(&rows);
        let mut seen = HashSet::new();
        let names: Vec<&str> = features
            .iter()
            .copied()
            .filter(|n| seen.insert(*n))
            .collect();
        let missing: Vec<String> = names
            .iter()
            .filter(|n| !best.contains_key(**n))
            .map(|n| n.to_string())
            .collect();
        if !missing.is_empty() {
            return Err(Error::MissingFeatures(missing));
        }
        let chosen: Vec<&FeatureRow> = names.iter().map(|n| &rows[best[*n]]).collect();
        let bad: Vec<(String, String)> = chosen
            .iter()
            .map(|r| (r.name.clone(), r.signal_type.clone().unwrap_or_default()))
            .filter(|(_, t)| !CONSOLIDATABLE_SIGNAL_TYPES.contains(&t.as_str()))
            .collect();
        if !bad.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "only features and targets can be consolidated; non-consolidatable signal \
                 types: {bad:?}"
            )));
        }
        let mut addrs: Vec<&str> = vec![];
        for r in &chosen {
            let addr = r
                .table_addr
                .as_deref()
                .ok_or_else(|| integrity("feature_metadata", format!("{} has no table", r.name)))?;
            if !addrs.contains(&addr) {
                addrs.push(addr);
            }
        }
        if addrs.len() <= 1 {
            return Ok(None);
        }

        let coverages = coverage_values(&tm)?;
        let mut values = vec![];
        for addr in &addrs {
            values.push(
                coverages
                    .get(*addr)
                    .ok_or_else(|| integrity(addr, "the table has no table metadata row"))?
                    .clone(),
            );
        }
        let converter = RowConverter::new(vec![SortField::new(values[0].data_type().clone())])?;
        let encoded = converter.convert_columns(&[arrow::compute::concat(
            &values.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
        )?])?;
        let newest = (0..encoded.num_rows())
            .max_by(|&a, &b| encoded.row(a).cmp(&encoded.row(b)))
            .unwrap();
        let coverage = values[newest].clone();

        let index = self.table(&self.options.index_path).await?.schema();
        let mut fields: Vec<Field> = self
            .sort_keys
            .iter()
            .map(|k| {
                index
                    .field_with_name(k)
                    .map(|f| Field::new(k, f.data_type().clone(), true))
            })
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| integrity(&self.options.index_path, e.to_string()))?;
        let mut schemas: HashMap<&str, SchemaRef> = HashMap::new();
        for addr in &addrs {
            schemas.insert(addr, self.table(addr).await?.schema());
        }
        for r in &chosen {
            let addr = r.table_addr.as_deref().unwrap();
            let field = schemas[addr]
                .field_with_name(&r.name)
                .map_err(|e| integrity(addr, e.to_string()))?;
            let metadata: HashMap<String, String> = field
                .metadata()
                .iter()
                .filter(|(k, _)| *k == COMMENT || *k == TAGS)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            fields
                .push(Field::new(&r.name, field.data_type().clone(), true).with_metadata(metadata));
        }
        let selected = chosen
            .iter()
            .map(|r| SelectedFeature {
                name: r.name.clone(),
                table_addr: r.table_addr.clone().unwrap(),
                signal_type: r.signal_type.clone().unwrap_or_default(),
                feature_params: r.feature_params.clone(),
            })
            .collect();
        let base = take_record_batch(
            &fm,
            &UInt32Array::from_iter_values(chosen.iter().map(|r| r.row as u32)),
        )?;
        Ok(Some(Consolidation {
            selected,
            coverage,
            schema: Arc::new(Schema::new(fields)),
            base,
            tm: tm_table,
            fm: fm_table,
        }))
    }
}

/// What [`Lake::consolidate`] writes, planned before anything is read or written.
struct Consolidation {
    selected: Vec<SelectedFeature>,
    /// Coverage of the new table: the newest coverage of the tables merged.
    coverage: ArrayRef,
    /// The sort keys, then the features with their comments and tags.
    schema: SchemaRef,
    /// The latest feature metadata row of each feature, in `selected` order.
    base: RecordBatch,
    /// The table and feature metadata tables, at the versions read.
    tm: Box<dyn StoredTable>,
    fm: Box<dyn StoredTable>,
}
