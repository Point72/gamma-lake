//! Registration of runtime-computed features.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, PoisonError};

use arrow::array::{ArrayRef, Int64Array, StringArray};

use crate::error::{Error, Result};
use crate::lake::{feature_rows, Lake};
use crate::lock::write_lock;
use crate::runtime::block_on;
use crate::write::{assemble, RUNTIME_COMPUTED};

/// A feature computed at read time from other features.
///
/// The core stores `expression` without interpreting it. The Python package stores polars
/// `Expr.meta.serialize(format="json")` and evaluates it over the `roots` features.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expression {
    pub name: String,
    pub expression: String,
    /// The features `expression` reads.
    pub roots: Vec<String>,
}

impl Lake {
    /// Registers `expressions` as runtime-computed features owned by `owner`, in one commit.
    ///
    /// Each name must be new to the lake, unique in `expressions` and not a sort key. Every
    /// root must be a registered feature or another of `expressions`, other than the
    /// expression itself, and `expressions` must not depend on each other in a cycle.
    pub fn add_expressions(&self, expressions: &[Expression], owner: &str) -> Result<()> {
        if expressions.is_empty() {
            return Ok(());
        }
        let lock = write_lock(self.store.location());
        let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        block_on(self.add_expressions_locked(expressions, owner))?
    }

    async fn add_expressions_locked(&self, expressions: &[Expression], owner: &str) -> Result<()> {
        let path = &self.options.feature_metadata_path;
        let registered: HashSet<String> = feature_rows(&self.read_batch(path).await?)?
            .into_iter()
            .map(|r| r.name)
            .collect();
        check(expressions, &registered, self.sort_keys())?;

        let table = self.table(path).await?;
        let schema = table.schema();
        let n = expressions.len();
        let strings = |values: Vec<&str>| -> ArrayRef { Arc::new(StringArray::from(values)) };
        let columns: HashMap<&str, ArrayRef> = HashMap::from([
            (
                "feature_name",
                strings(expressions.iter().map(|e| e.name.as_str()).collect()),
            ),
            (
                "version",
                Arc::new(Int64Array::from(vec![0; n])) as ArrayRef,
            ),
            ("table_addr", strings(vec!["n/a"; n])),
            ("owner", strings(vec![owner; n])),
            ("signal_type", strings(vec![RUNTIME_COMPUTED; n])),
            (
                "feature_params",
                strings(expressions.iter().map(|e| e.expression.as_str()).collect()),
            ),
        ]);
        table.append(assemble(&schema, columns, n, path)?).await
    }
}

/// Checks `expressions` against the `registered` feature names before anything is written.
fn check(
    expressions: &[Expression],
    registered: &HashSet<String>,
    sort_keys: &[String],
) -> Result<()> {
    let invalid = |message: String| Err(Error::InvalidArgument(message));
    let mut names = HashSet::new();
    for e in expressions {
        if e.name.is_empty() {
            return invalid("a runtime-computed feature needs a name".into());
        }
        if !names.insert(e.name.as_str()) {
            return invalid(format!("feature {:?} is registered twice", e.name));
        }
        if sort_keys.contains(&e.name) {
            return invalid(format!("feature {:?} is a sort key of the lake", e.name));
        }
        if registered.contains(&e.name) {
            return invalid(format!(
                "feature {:?} is already registered and cannot be re-registered",
                e.name
            ));
        }
        if e.roots.contains(&e.name) {
            return invalid(format!("feature {:?} reads itself", e.name));
        }
    }
    let missing: Vec<&str> = expressions
        .iter()
        .flat_map(|e| &e.roots)
        .map(String::as_str)
        .filter(|r| !registered.contains(*r) && !names.contains(r))
        .collect();
    if !missing.is_empty() {
        return invalid(format!(
            "the expressions read features that are not yet stored in or computed by the lake: \
             {missing:?}"
        ));
    }
    if let Some(name) = cycle(expressions) {
        return invalid(format!(
            "feature {name:?} depends on itself through other features"
        ));
    }
    Ok(())
}

/// A name in a dependency cycle among `expressions`, if there is one.
fn cycle(expressions: &[Expression]) -> Option<&str> {
    let names: HashSet<&str> = expressions.iter().map(|e| e.name.as_str()).collect();
    // Each name with the request-local roots it still waits on. Names waiting on nothing are
    // removed until none are left; any name remaining is on or behind a cycle.
    let mut pending: HashMap<&str, HashSet<&str>> = expressions
        .iter()
        .map(|e| {
            let roots = e.roots.iter().map(String::as_str);
            (
                e.name.as_str(),
                roots.filter(|r| names.contains(r)).collect(),
            )
        })
        .collect();
    loop {
        let ready: Vec<&str> = pending
            .iter()
            .filter(|(_, roots)| roots.is_empty())
            .map(|(name, _)| *name)
            .collect();
        if ready.is_empty() {
            break;
        }
        for done in ready {
            pending.remove(done);
            for roots in pending.values_mut() {
                roots.remove(done);
            }
        }
    }
    expressions
        .iter()
        .map(|e| e.name.as_str())
        .find(|n| pending.contains_key(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expr(name: &str, roots: &[&str]) -> Expression {
        Expression {
            name: name.into(),
            expression: "{}".into(),
            roots: roots.iter().map(|r| r.to_string()).collect(),
        }
    }

    fn registered() -> HashSet<String> {
        HashSet::from(["a".to_string(), "b".to_string()])
    }

    fn message(expressions: &[Expression]) -> String {
        match check(expressions, &registered(), &["date".to_string()]) {
            Err(Error::InvalidArgument(m)) => m,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn accepts_roots_registered_or_in_the_request() {
        let ok = [expr("c", &["a", "d"]), expr("d", &["b"]), expr("e", &[])];
        check(&ok, &registered(), &["date".to_string()]).unwrap();
    }

    #[test]
    fn rejects_invalid_expressions() {
        assert!(message(&[expr("", &[])]).contains("needs a name"));
        assert!(message(&[expr("c", &[]), expr("c", &[])]).contains("twice"));
        assert!(message(&[expr("date", &[])]).contains("sort key"));
        assert!(message(&[expr("a", &[])]).contains("already registered"));
        assert!(message(&[expr("c", &["c"])]).contains("reads itself"));
        assert!(message(&[expr("c", &["z"])]).contains("[\"z\"]"));
        let cyclic = [expr("c", &["d"]), expr("d", &["e"]), expr("e", &["c"])];
        assert!(message(&cyclic).contains("depends on itself"));
        for _ in 0..20 {
            // Independent names next to a cycle, in every hash order.
            let mixed = [
                expr("x", &[]),
                expr("y", &[]),
                expr("c", &["x", "d"]),
                expr("d", &["c"]),
            ];
            assert!(message(&mixed).contains("depends on itself"));
        }
    }
}
