//! Tokenisation shared by lexical retrieval, memory and loop detection.

use std::collections::BTreeMap;

/// Words that carry no retrieval signal in code or conversation text.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "not", "but", "you",
    "your", "can", "all", "any", "its", "has", "have", "how", "what", "when", "where", "which",
    "there", "will", "should", "would", "could", "please", "does", "did", "get", "set", "let",
];

/// Splits text into lowercase index terms.
///
/// Identifiers are split on non-alphanumeric characters and on camelCase
/// boundaries. The whole lowercase identifier is kept too, so exact-name
/// queries still match. Short tokens, pure numbers and stopwords are dropped.
pub fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let parts = split_camel(word);
        if parts.len() > 1 {
            push_term(&mut out, &word.to_lowercase());
        }
        for part in parts {
            push_term(&mut out, &part);
        }
    }
    out
}

/// Term frequencies for `text`, as used by the chunk index.
pub fn term_freq(text: &str) -> BTreeMap<String, u32> {
    let mut tf = BTreeMap::new();
    for t in terms(text) {
        *tf.entry(t).or_insert(0) += 1;
    }
    tf
}

/// Distinct terms of `text`, sorted.
pub fn term_set(text: &str) -> std::collections::BTreeSet<String> {
    terms(text).into_iter().collect()
}

fn push_term(out: &mut Vec<String>, t: &str) {
    if t.chars().count() < 2 || t.chars().all(|c| c.is_ascii_digit()) || STOPWORDS.contains(&t) {
        return;
    }
    out.push(light_stem(t));
}

/// Strips a plural `s` from words of four or more letters ("refunds" -> "refund").
/// Deliberately conservative: words ending in `ss`, `us` or `is` are kept as they are.
fn light_stem(t: &str) -> String {
    let plural = t.chars().count() >= 4
        && t.ends_with('s')
        && !t.ends_with("ss")
        && !t.ends_with("us")
        && !t.ends_with("is");
    if plural {
        t[..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

/// Splits `word` at camelCase and acronym boundaries, lowercasing each part.
fn split_camel(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut parts = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let boundary = i > 0
            && c.is_uppercase()
            && (chars[i - 1].is_lowercase()
                || chars[i - 1].is_ascii_digit()
                || (chars[i - 1].is_uppercase()
                    && chars.get(i + 1).is_some_and(|n| n.is_lowercase())));
        if boundary && !cur.is_empty() {
            parts.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts.into_iter().map(|p| p.to_lowercase()).collect()
}

/// Normalises a message or log line into a loop-detection signature: digits
/// become `#`, whitespace collapses and the result is lowercased and capped.
pub fn normalize_signature(text: &str) -> String {
    let mut out = String::new();
    let mut last_space = false;
    for c in text.trim().chars() {
        let mapped = if c.is_ascii_digit() {
            '#'
        } else {
            c.to_ascii_lowercase()
        };
        if mapped.is_whitespace() {
            if !last_space {
                out.push(' ');
            }
            last_space = true;
        } else {
            out.push(mapped);
            last_space = false;
        }
        if out.chars().count() >= 200 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_snake_and_camel_identifiers() {
        let t = terms("reconcile_refunds ledgerEntry");
        assert!(t.contains(&"reconcile".to_string()));
        assert!(t.contains(&"refund".to_string()), "plural is stemmed");
        assert!(!t.contains(&"refunds".to_string()));
        assert!(t.contains(&"ledger".to_string()));
        assert!(t.contains(&"entry".to_string()));
        assert!(t.contains(&"ledgerentry".to_string()));
    }

    #[test]
    fn acronym_boundary() {
        assert_eq!(split_camel("HTTPServer"), vec!["http", "server"]);
    }

    #[test]
    fn drops_stopwords_short_and_numeric() {
        let t = terms("the a 42 and");
        assert!(t.is_empty(), "{t:?}");
    }

    #[test]
    fn light_stem_is_conservative() {
        assert_eq!(light_stem("ledgers"), "ledger");
        assert_eq!(light_stem("status"), "status");
        assert_eq!(light_stem("class"), "class");
        assert_eq!(light_stem("is"), "is");
    }

    #[test]
    fn keeps_alnum_mixed_tokens() {
        assert!(terms("v2").contains(&"v2".to_string()));
    }

    #[test]
    fn term_freq_counts_repeats() {
        let tf = term_freq("refund refund ledger");
        assert_eq!(tf.get("refund"), Some(&2));
        assert_eq!(tf.get("ledger"), Some(&1));
    }

    #[test]
    fn signature_masks_digits_and_whitespace() {
        assert_eq!(
            normalize_signature("Error  at line 42\n"),
            "error at line ##"
        );
    }

    #[test]
    fn signature_is_capped() {
        let long = "x".repeat(500);
        assert_eq!(normalize_signature(&long).chars().count(), 200);
    }
}
