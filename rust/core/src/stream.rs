//! Record batch streams returned by the core.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::vec;

use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt, TryStreamExt};

use crate::runtime;

/// An asynchronous stream of Arrow record batches sharing one schema.
pub struct AsyncBatchStream {
    schema: SchemaRef,
    inner: Pin<Box<dyn Stream<Item = crate::Result<RecordBatch>> + Send>>,
}

impl AsyncBatchStream {
    pub(crate) fn from_plan(stream: SendableRecordBatchStream) -> Self {
        Self {
            schema: stream.schema(),
            inner: Box::pin(stream.map_err(crate::Error::from)),
        }
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Stream for AsyncBatchStream {
    type Item = crate::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

enum Source {
    Plan(SendableRecordBatchStream),
    Memory(vec::IntoIter<RecordBatch>),
}

/// A blocking iterator of record batches sharing one schema.
///
/// The stream owns everything it needs, so it may outlive the [`crate::Lake`] that
/// produced it. After the first error it yields no more batches.
pub struct BatchStream {
    schema: SchemaRef,
    source: Option<Source>,
}

static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Number of [`BatchStream`]s not yet dropped, for leak tests in the bindings.
#[doc(hidden)]
pub fn live_streams() -> usize {
    LIVE.load(Ordering::SeqCst)
}

impl Drop for BatchStream {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

impl BatchStream {
    pub(crate) fn from_plan(stream: SendableRecordBatchStream) -> Self {
        LIVE.fetch_add(1, Ordering::SeqCst);
        Self {
            schema: stream.schema(),
            source: Some(Source::Plan(stream)),
        }
    }

    /// A stream of `batches`, which must have `schema`.
    pub fn from_batches(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        LIVE.fetch_add(1, Ordering::SeqCst);
        Self {
            schema,
            source: Some(Source::Memory(batches.into_iter())),
        }
    }

    fn fail(&mut self, message: String) -> Option<Result<RecordBatch, ArrowError>> {
        self.source = None;
        Some(Err(ArrowError::ExternalError(
            message.replace('\0', "\\0").into(),
        )))
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

impl Iterator for BatchStream {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.source.as_mut()? {
            Source::Memory(batches) => return batches.next().map(Ok),
            Source::Plan(stream) => {
                catch_unwind(AssertUnwindSafe(|| runtime::block_on(stream.next())))
            }
        };
        match next {
            Ok(Ok(Some(Ok(batch)))) => Some(Ok(batch)),
            Ok(Ok(Some(Err(e)))) => self.fail(e.to_string()),
            Ok(Ok(None)) => {
                self.source = None;
                None
            }
            Ok(Err(e)) => self.fail(e.to_string()),
            Err(payload) => self.fail(format!(
                "gamma-lake stream panicked: {}",
                panic_message(payload.as_ref())
            )),
        }
    }
}

impl RecordBatchReader for BatchStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StructArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
    use arrow::ffi_stream::FFI_ArrowArrayStream;
    use datafusion::error::{DataFusionError, Result as DfResult};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use std::ffi::CStr;
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]))
    }

    fn batch() -> RecordBatch {
        RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(vec![7]))]).unwrap()
    }

    fn exported(
        items: impl Stream<Item = DfResult<RecordBatch>> + Send + 'static,
    ) -> FFI_ArrowArrayStream {
        let stream = RecordBatchStreamAdapter::new(schema(), items);
        FFI_ArrowArrayStream::new(Box::new(BatchStream::from_plan(Box::pin(stream))))
    }

    /// Calls `get_next`, returning the status and the batch if one was produced.
    fn next(stream: &mut FFI_ArrowArrayStream) -> (i32, Option<FFI_ArrowArray>) {
        let mut out = FFI_ArrowArray::empty();
        let rc = unsafe { (stream.get_next.unwrap())(stream, &mut out) };
        (rc, (!out.is_released()).then_some(out))
    }

    fn last_error(stream: &mut FFI_ArrowArrayStream) -> String {
        let message = unsafe { CStr::from_ptr((stream.get_last_error.unwrap())(stream)) };
        message.to_str().unwrap().to_string()
    }

    fn release(mut stream: FFI_ArrowArrayStream) {
        unsafe { (stream.release.unwrap())(&mut stream) };
    }

    #[test]
    fn errors_are_reported_then_the_stream_ends() {
        let items = futures::stream::iter(vec![
            Ok(batch()),
            Err(DataFusionError::Execution("bad\0value".into())),
            Ok(batch()),
        ]);
        let mut stream = exported(items);
        let (rc, kept) = next(&mut stream);
        assert_eq!(rc, 0);
        let (rc, none) = next(&mut stream);
        assert_ne!(rc, 0);
        assert!(none.is_none());
        assert!(last_error(&mut stream).contains("bad\\0value"));
        let (rc, end) = next(&mut stream);
        assert_eq!(rc, 0);
        assert!(end.is_none());
        release(stream);

        let schema = FFI_ArrowSchema::try_from(schema().as_ref()).unwrap();
        let data = unsafe { from_ffi(kept.unwrap(), &schema) }.unwrap();
        let batch = RecordBatch::from(StructArray::from(data));
        assert_eq!(batch, super::tests::batch());
    }

    #[test]
    fn panics_while_polling_are_contained() {
        let items = futures::stream::once(async { panic!("polling boom") });
        let mut stream = exported(items);
        assert_ne!(next(&mut stream).0, 0);
        assert!(last_error(&mut stream).contains("polling boom"));
        assert_eq!(next(&mut stream).0, 0);
        release(stream);
    }

    #[test]
    fn panics_in_spawned_tasks_are_contained() {
        // DataFusion re-raises panics from its spawned tasks in the polling thread.
        let items = futures::stream::once(async {
            match tokio::spawn(async { panic!("task boom") }).await {
                Ok(never) => never,
                Err(e) => std::panic::resume_unwind(e.into_panic()),
            }
        });
        let mut stream = exported(items);
        assert_ne!(next(&mut stream).0, 0);
        assert!(last_error(&mut stream).contains("task boom"));
        release(stream);
    }
}
