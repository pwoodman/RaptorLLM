//! Long answers written as parallel sections ("skeleton of thought"). One request plans the
//! answer (a `Facts:` line every section must keep, then a numbered outline); then every section
//! is written at the same time in its own server slot and shown in order. The answer arrives
//! sooner than one long reply because a slot-parallel batch costs about as much as one token.
//!
//! Measured on Nanbeige4.3-33B IQ4_XS (RX 9060 XT, llama-server `-np 8 -c 8192 -ub 64 -b 64`,
//! 2026-11-14): whole answers at 44.7 (story, 3,770 tokens), 33.7 (explanation) and 31.6 tok/s
//! (guide) wall-clock, vs about 11-14 tok/s for one reply. The output is structured differently
//! from a single reply: sections are written without seeing each other, so each tends to end
//! with its own wrap-up sentence. Without the `Facts:` line a story changed its heroine's name
//! between sections; with it the names held. Fewer slots than sections makes it slower.
//!
//! This module is pure: prompts, plan parsing and in-order emission. The requests are in
//! `commands::chat`.

/// Appended to the user's last message for the planning request.
pub const PLAN_INSTRUCTION: &str = "\n\nFirst, plan the answer. Line 1 starts with 'Facts:' and \
lists, in at most 25 words, the fixed details every section must share (for a story: character \
names, setting, point of view, tense; otherwise: audience and key terms). Then a numbered outline \
of 8 sections, one line each (6-12 words) saying exactly what that section covers or what happens \
in it, with no overlap between sections. No other text.";

/// Token limit of the planning request (a Facts line and 8 short lines).
pub const PLAN_MAX_TOKENS: u32 = 170;

/// Token limit of one section. With "2-3 detailed paragraphs" some sections in the app ran past
/// 450 and 600 tokens and stopped mid-word, so the cap is 700. Asking for a word count instead
/// made the model echo the instructions into the answer ("Final Version (Target: 200-300
/// words)") and repeat a section, so the prompt does not. The slowest section sets the total time.
pub const SECTION_MAX_TOKENS: u32 = 700;

/// Most sections written at once.
pub const MAX_SECTIONS: usize = 8;

/// The parsed plan: the shared facts and one line per section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub facts: String,
    pub points: Vec<String>,
}

impl Plan {
    /// The plan as the assistant's turn in each section's conversation.
    pub fn as_text(&self) -> String {
        let mut out = String::new();
        if !self.facts.is_empty() {
            out.push_str(&self.facts);
            out.push('\n');
        }
        out.push_str("Outline:\n");
        for (i, p) in self.points.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, p));
        }
        out
    }

    /// The request for section `i` (0-based).
    pub fn section_prompt(&self, i: usize) -> String {
        format!(
            "Write section {} of {} in full: \"{}\". 2-3 detailed paragraphs. Keep every fixed \
             detail from the Facts line exactly. Cover only what this outline line says (earlier \
             sections already happened, later ones have not). Start with the heading '## {}. \
             <short title>'.",
            i + 1,
            self.points.len(),
            self.points[i],
            i + 1
        )
    }
}

/// Reads a plan: an optional `Facts:` line and numbered lines (`1.` or `1)`). None when fewer
/// than two sections were planned, so the caller answers normally instead.
pub fn parse_plan(text: &str) -> Option<Plan> {
    let mut facts = String::new();
    let mut points = Vec::new();
    for line in text.lines() {
        let line = line.trim().trim_start_matches(['*', '-', '#', ' ']).trim();
        if facts.is_empty() && line.to_ascii_lowercase().starts_with("facts") {
            facts = line.replace("**", "").trim().to_string();
            continue;
        }
        let digits = line.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 || digits > 2 {
            continue;
        }
        let rest = &line[digits..];
        let Some(rest) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) else {
            continue;
        };
        let point = rest.replace("**", "").trim().to_string();
        if !point.is_empty() && points.len() < MAX_SECTIONS {
            points.push(point);
        }
    }
    (points.len() >= 2).then_some(Plan { facts, points })
}

/// Emits section text in order while sections finish out of order: section 0 streams live,
/// later sections are held until every earlier one has finished.
#[derive(Debug)]
pub struct InOrder {
    next: usize,
    held: Vec<String>,
    finished: Vec<bool>,
}

/// Put between sections.
pub const SEPARATOR: &str = "\n\n";

impl InOrder {
    pub fn new(sections: usize) -> Self {
        Self {
            next: 0,
            held: vec![String::new(); sections],
            finished: vec![false; sections],
        }
    }

    /// Text that arrived for section `i`; returns what can be shown now.
    pub fn push(&mut self, i: usize, text: &str) -> Option<String> {
        if i == self.next {
            Some(text.to_string())
        } else {
            self.held[i].push_str(text);
            None
        }
    }

    /// Section `i` finished; returns what can be shown now (held sections that are next in line).
    pub fn finish(&mut self, i: usize) -> Option<String> {
        self.finished[i] = true;
        let mut out = String::new();
        while self.next < self.finished.len() && self.finished[self.next] {
            self.next += 1;
            if self.next < self.held.len() {
                out.push_str(SEPARATOR);
                out.push_str(&std::mem::take(&mut self.held[self.next]));
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// True once every section has finished.
    pub fn done(&self) -> bool {
        self.next >= self.finished.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_keeps_the_facts_line_and_the_numbered_sections() {
        let text = "Facts: Mara, keeper of Stormcliff light; third person, past tense.\n\
                    1. The lonely rhythm of the light\n\
                    2) A bottle arrives at dawn\n\
                    **3.** The letter in her own hand\n\
                    Some chatter that is not a section";
        let plan = parse_plan(text).unwrap();
        assert!(plan.facts.starts_with("Facts: Mara"));
        assert_eq!(
            plan.points,
            [
                "The lonely rhythm of the light",
                "A bottle arrives at dawn",
                "The letter in her own hand"
            ]
        );
        assert!(plan.as_text().contains("Outline:\n1. The lonely rhythm"));
        let p = plan.section_prompt(1);
        assert!(p.starts_with("Write section 2 of 3 in full: \"A bottle arrives at dawn\""));
        assert!(p.contains("'## 2. <short title>'"));
    }

    #[test]
    fn fewer_than_two_sections_is_no_plan_and_more_than_eight_are_cut() {
        assert_eq!(parse_plan("Facts: none\n1. Only one"), None);
        assert_eq!(parse_plan("no outline at all"), None);
        let many: String = (1..=12).map(|i| format!("{i}. point {i}\n")).collect();
        let plan = parse_plan(&many).unwrap();
        assert_eq!(plan.points.len(), MAX_SECTIONS);
        assert_eq!(plan.facts, "");
        assert!(plan.as_text().starts_with("Outline:"));
        // A year is not a section number.
        assert_eq!(parse_plan("1999. a year\n2024. another"), None);
    }

    #[test]
    fn sections_are_shown_in_order_whatever_order_they_finish_in() {
        let mut o = InOrder::new(3);
        assert_eq!(
            o.push(0, "A"),
            Some("A".into()),
            "the first section streams live"
        );
        assert_eq!(o.push(2, "C"), None);
        assert_eq!(o.push(1, "B"), None);
        assert_eq!(o.finish(2), None, "section 2 waits for 0 and 1");
        assert_eq!(o.push(0, "a"), Some("a".into()));
        assert_eq!(
            o.finish(0),
            Some("\n\nB".into()),
            "1 is next, so what it has so far shows"
        );
        assert_eq!(o.push(1, "b"), Some("b".into()), "and it now streams live");
        assert!(!o.done());
        assert_eq!(o.finish(1), Some("\n\nC".into()), "2 had already finished");
        assert!(o.done());
    }
}
