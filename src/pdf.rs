//! Text of a PDF, read with the system's PDFKit.
use crate::{Result, invalid};
use objc2::AnyThread;
use objc2_foundation::NSData;
use objc2_pdf_kit::PDFDocument;

/// Returns the text and the page count. A scanned PDF has no text, which is
/// an error here: an empty attachment would look like a successful one.
pub fn extract(bytes: &[u8]) -> Result<(String, usize)> {
    objc2::rc::autoreleasepool(|_| {
        let data = NSData::with_bytes(bytes);
        let document = unsafe { PDFDocument::initWithData(PDFDocument::alloc(), &data) }
            .ok_or_else(|| invalid("this file is not a readable PDF"))?;
        let text = unsafe { document.string() }.map_or(String::new(), |s| s.to_string());
        if text.trim().is_empty() {
            return Err(invalid(
                "this PDF has no text to read; it may be a scan of images",
            ));
        }
        Ok((text, unsafe { document.pageCount() }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn other_files_are_refused() {
        assert!(extract(b"not a pdf").is_err());
    }
}
