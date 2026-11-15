//! Packing: decides which candidates fit the budget, elides where structure allows,
//! and records why each item was kept or dropped.
//!
//! Rules, in order:
//! 1. Protected candidates are always included in full, even over budget.
//! 2. Unprotected candidates fill their category cap, best score first.
//! 3. Unused cap is spilled to whatever candidates are still waiting, best score first.
//! 4. A candidate that does not fit whole may be elided (signature plus focus lines).

use crate::budget::{allocate, Category, TaskKind};
use crate::provenance::{CategoryUsage, ExcludedItem, LineRange};
use raptor_core::{ChatMessage, Role, TokenCounter};
use std::collections::BTreeMap;

/// One thing that could go into the context.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub source: String,
    pub category: Category,
    /// Protected candidates are never dropped or elided.
    pub protected: bool,
    pub hash: Option<String>,
    pub range: Option<LineRange>,
    pub score: f64,
    pub why: String,
    /// Heading rendered above the body, e.g. `### src/a.rs:10-40`. Empty for history.
    pub heading: String,
    pub body: String,
    /// Whether the body may be elided to fit.
    pub elidable: bool,
    /// Terms that must survive elision (usually the query terms).
    pub focus: Vec<String>,
    pub recover_handle: Option<String>,
    /// Conversation index, for history candidates.
    pub message: Option<usize>,
    /// Role, for history candidates.
    pub role: Option<Role>,
}

impl Candidate {
    /// A candidate with the given identity and body. Other fields start empty and can be set.
    pub fn new(
        id: impl Into<String>,
        category: Category,
        protected: bool,
        body: impl Into<String>,
    ) -> Self {
        let id = id.into();
        Self {
            source: id.clone(),
            id,
            category,
            protected,
            hash: None,
            range: None,
            score: 0.0,
            why: String::new(),
            heading: String::new(),
            body: body.into(),
            elidable: false,
            focus: Vec::new(),
            recover_handle: None,
            message: None,
            role: None,
        }
    }
}

/// A candidate that made it into the context, with the text that will be rendered.
#[derive(Debug, Clone)]
pub struct Selected {
    pub cand: Candidate,
    /// Rendered text: heading plus (possibly elided) body.
    pub text: String,
    pub tokens: usize,
    pub truncated: bool,
}

/// Result of [`pack`].
#[derive(Debug, Clone)]
pub struct PackResult {
    pub selected: Vec<Selected>,
    pub excluded: Vec<ExcludedItem>,
    pub utilization: BTreeMap<Category, CategoryUsage>,
    pub used: usize,
    pub over_budget_by: usize,
}

/// Renders heading and body as they will appear in the context.
pub fn render(cand: &Candidate, body: &str) -> String {
    if cand.heading.is_empty() {
        body.to_string()
    } else {
        format!("{}\n{}", cand.heading, body)
    }
}

#[derive(Clone, Copy)]
enum ElideLevel {
    /// Keep the first and last line and every line that mentions a focus term.
    Focus,
    /// Keep only the first and last line.
    Signature,
}

/// Structure-aware elision: removed runs are replaced by a marker line, never cut mid-line.
fn elide(body: &str, focus: &[String], level: ElideLevel) -> String {
    let lines: Vec<&str> = body.lines().collect();
    let n = lines.len();
    if n <= 2 {
        return body.to_string();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    if matches!(level, ElideLevel::Focus) {
        for (i, line) in lines.iter().enumerate() {
            if crate::text::terms(line).iter().any(|t| focus.contains(t)) {
                keep[i] = true;
            }
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < n {
        if keep[i] {
            out.push(lines[i].to_string());
            i += 1;
            continue;
        }
        let start = i;
        while i < n && !keep[i] {
            i += 1;
        }
        out.push(format!("... {} lines elided ...", i - start));
    }
    out.join("\n")
}

/// Fits `cand` into `limit` tokens: whole if possible, otherwise elided at the richest level that fits.
fn fit(
    cand: &Candidate,
    limit: usize,
    counter: &dyn TokenCounter,
) -> Option<(String, usize, bool)> {
    let full = render(cand, &cand.body);
    let tokens = counter.count(&full);
    if tokens <= limit {
        return Some((full, tokens, false));
    }
    if !cand.elidable {
        return None;
    }
    for level in [ElideLevel::Focus, ElideLevel::Signature] {
        let text = render(cand, &elide(&cand.body, &cand.focus, level));
        let t = counter.count(&text);
        if t <= limit {
            return Some((text, t, true));
        }
    }
    None
}

fn accept(cand: Candidate, text: String, tokens: usize, truncated: bool) -> Selected {
    Selected {
        cand,
        text,
        tokens,
        truncated,
    }
}

/// Packs `candidates` into `available` tokens. `pre_excluded` are items the caller already
/// dropped (for example stale memory); they are passed through to the result unchanged.
pub fn pack(
    candidates: Vec<Candidate>,
    pre_excluded: Vec<ExcludedItem>,
    available: usize,
    kind: TaskKind,
    counter: &dyn TokenCounter,
) -> PackResult {
    let (protected, mut open): (Vec<Candidate>, Vec<Candidate>) =
        candidates.into_iter().partition(|c| c.protected);
    let mut selected: Vec<Selected> = Vec::new();

    let mut protected_used = 0usize;
    for cand in protected {
        let text = render(&cand, &cand.body);
        let tokens = counter.count(&text);
        protected_used += tokens;
        selected.push(accept(cand, text, tokens, false));
    }
    let over_budget_by = protected_used.saturating_sub(available);
    let pool = available.saturating_sub(protected_used);
    let caps = allocate(kind, pool);

    open.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    let mut cat_used: BTreeMap<Category, usize> = BTreeMap::new();
    let mut pool_used = 0usize;
    let mut deferred: Vec<Candidate> = Vec::new();
    for cand in open {
        let cap = caps.get(&cand.category).copied().unwrap_or(0);
        let used_cat = cat_used.get(&cand.category).copied().unwrap_or(0);
        let limit = cap
            .saturating_sub(used_cat)
            .min(pool.saturating_sub(pool_used));
        match fit(&cand, limit, counter) {
            Some((text, tokens, truncated)) => {
                *cat_used.entry(cand.category).or_insert(0) += tokens;
                pool_used += tokens;
                selected.push(accept(cand, text, tokens, truncated));
            }
            None => deferred.push(cand),
        }
    }

    let mut excluded: Vec<ExcludedItem> = pre_excluded;
    for cand in deferred {
        let limit = pool.saturating_sub(pool_used);
        match fit(&cand, limit, counter) {
            Some((text, tokens, truncated)) => {
                *cat_used.entry(cand.category).or_insert(0) += tokens;
                pool_used += tokens;
                selected.push(accept(cand, text, tokens, truncated));
            }
            None => {
                let full_tokens = counter.count(&render(&cand, &cand.body));
                let cap = caps.get(&cand.category).copied().unwrap_or(0);
                let why = format!(
                    "over budget: needs {full_tokens} tokens; {} category tokens left of {cap}, {} pool tokens left",
                    cap.saturating_sub(cat_used.get(&cand.category).copied().unwrap_or(0)),
                    pool.saturating_sub(pool_used)
                );
                excluded.push(ExcludedItem {
                    id: cand.id.clone(),
                    source: cand.source.clone(),
                    category: cand.category,
                    tokens: full_tokens,
                    why,
                    recover_handle: cand.recover_handle.clone(),
                });
            }
        }
    }

    selected.sort_by(|a, b| {
        a.cand
            .category
            .cmp(&b.cand.category)
            .then_with(|| b.cand.score.total_cmp(&a.cand.score))
            .then_with(|| a.cand.id.cmp(&b.cand.id))
    });
    let used = selected.iter().map(|s| s.tokens).sum();
    let mut utilization = BTreeMap::new();
    for cat in Category::ALL {
        let budget = caps.get(&cat).copied();
        utilization.insert(cat, CategoryUsage { used: 0, budget });
    }
    for s in &selected {
        if let Some(u) = utilization.get_mut(&s.cand.category) {
            u.used += s.tokens;
        }
    }
    PackResult {
        selected,
        excluded,
        utilization,
        used,
        over_budget_by,
    }
}

/// Builds the returned messages: one system message with every non-history selected
/// item grouped by category, followed by the kept history turns in their original order.
pub fn assemble_messages(selected: &[Selected]) -> Vec<ChatMessage> {
    let mut sections: Vec<String> = Vec::new();
    for cat in Category::ALL {
        if cat == Category::History {
            continue;
        }
        let parts: Vec<&str> = selected
            .iter()
            .filter(|s| s.cand.category == cat)
            .map(|s| s.text.as_str())
            .collect();
        if !parts.is_empty() {
            sections.push(format!("## {}\n\n{}", cat.heading(), parts.join("\n\n")));
        }
    }
    let mut out = Vec::new();
    if !sections.is_empty() {
        out.push(ChatMessage::new(Role::System, sections.join("\n\n")));
    }
    let mut history: Vec<(usize, &Selected)> = selected
        .iter()
        .filter(|s| s.cand.category == Category::History)
        .filter_map(|s| s.cand.message.map(|m| (m, s)))
        .collect();
    history.sort_by_key(|(m, _)| *m);
    for (_, s) in history {
        out.push(ChatMessage::new(
            s.cand.role.unwrap_or(Role::User),
            s.text.clone(),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use raptor_core::EstimatingCounter;

    fn cand(id: &str, cat: Category, protected: bool, score: f64, body: &str) -> Candidate {
        let mut c = Candidate::new(id, cat, protected, body);
        c.score = score;
        c.why = "test".into();
        c
    }

    fn tokens(s: &str) -> usize {
        EstimatingCounter.count(s)
    }

    #[test]
    fn protected_items_survive_a_tiny_budget() {
        let big = "x".repeat(400);
        let packed = pack(
            vec![
                cand("task", Category::Task, true, 0.0, &big),
                cand("ret", Category::RetrievedCode, false, 1.0, "small"),
            ],
            vec![],
            10,
            TaskKind::Feature,
            &EstimatingCounter,
        );
        assert!(packed.selected.iter().any(|s| s.cand.id == "task"));
        assert!(packed.over_budget_by > 0);
        assert!(packed.excluded.iter().any(|e| e.id == "ret"));
    }

    #[test]
    fn unprotected_fill_prefers_higher_score_when_only_one_fits() {
        let body = "word ".repeat(20);
        let t = tokens(&body);
        let cands = vec![
            cand("low", Category::RetrievedCode, false, 0.1, &body),
            cand("high", Category::RetrievedCode, false, 0.9, &body),
        ];
        // The pool holds exactly one candidate, so the higher score must win.
        let packed = pack(cands, vec![], t, TaskKind::Feature, &EstimatingCounter);
        let ids: Vec<&str> = packed.selected.iter().map(|s| s.cand.id.as_str()).collect();
        assert_eq!(ids, vec!["high"]);
        assert!(packed
            .excluded
            .iter()
            .any(|e| e.id == "low" && e.why.starts_with("over budget")));
        assert!(packed.used <= t);
    }

    #[test]
    fn excluded_items_carry_reason_and_handle() {
        let body = "y ".repeat(500);
        let mut c = cand("chunk:a", Category::RetrievedCode, false, 1.0, &body);
        c.recover_handle = Some("chunk:a:1-2:abc".into());
        let packed = pack(vec![c], vec![], 20, TaskKind::Feature, &EstimatingCounter);
        let e = packed.excluded.first().expect("excluded");
        assert!(!e.why.is_empty());
        assert_eq!(e.recover_handle.as_deref(), Some("chunk:a:1-2:abc"));
    }

    #[test]
    fn elision_keeps_signature_and_focus_lines_and_marks_gaps() {
        let body = "fn refund() {\n    a();\n    b();\n    issue_refund(x);\n    c();\n}";
        let out = elide(body, &["refund".to_string()], ElideLevel::Focus);
        assert!(out.starts_with("fn refund() {"));
        assert!(out.contains("issue_refund(x);"));
        assert!(out.contains("lines elided"));
        assert!(out.ends_with('}'));
    }

    #[test]
    fn signature_level_is_smaller_than_focus_level() {
        let body = (0..30)
            .map(|i| format!("line {i} refund"))
            .collect::<Vec<_>>()
            .join("\n");
        let focus = elide(&body, &["refund".to_string()], ElideLevel::Focus);
        let sig = elide(&body, &["refund".to_string()], ElideLevel::Signature);
        assert!(sig.len() < focus.len());
    }

    #[test]
    fn elided_candidate_is_marked_truncated() {
        let body = (0..80)
            .map(|i| format!("filler number {i} with padding words"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut c = cand("chunk:x", Category::RetrievedCode, false, 1.0, &body);
        c.elidable = true;
        c.focus = vec!["zzzz".into()];
        let full = tokens(&body);
        let packed = pack(
            vec![c],
            vec![],
            full / 2 + 40,
            TaskKind::Feature,
            &EstimatingCounter,
        );
        let s = packed.selected.first().expect("selected elided");
        assert!(s.truncated);
        assert!(s.text.contains("lines elided"));
    }

    #[test]
    fn non_elidable_history_is_dropped_whole() {
        let body = "turn ".repeat(300);
        let mut c = cand("history:1", Category::History, false, 0.5, &body);
        c.message = Some(1);
        c.role = Some(Role::User);
        let packed = pack(vec![c], vec![], 30, TaskKind::Feature, &EstimatingCounter);
        assert!(packed.selected.is_empty());
        assert_eq!(packed.excluded.len(), 1);
    }

    #[test]
    fn spill_uses_unused_category_budget() {
        // Only history has candidates: it may use the full pool, not just its 25% share.
        let body = "turn ".repeat(10);
        let t = tokens(&body);
        let cands: Vec<Candidate> = (0..6)
            .map(|i| {
                let mut c = cand(
                    &format!("history:{i}"),
                    Category::History,
                    false,
                    1.0 - i as f64 / 10.0,
                    &body,
                );
                c.message = Some(i);
                c.role = Some(Role::User);
                c
            })
            .collect();
        let packed = pack(
            cands,
            vec![],
            t * 6 + 5,
            TaskKind::Feature,
            &EstimatingCounter,
        );
        assert_eq!(
            packed.selected.len(),
            6,
            "spill should reach the whole pool"
        );
    }

    #[test]
    fn utilization_lists_every_category() {
        let packed = pack(vec![], vec![], 100, TaskKind::Debug, &EstimatingCounter);
        assert_eq!(packed.utilization.len(), Category::ALL.len());
        assert_eq!(packed.utilization[&Category::Task].budget, None);
        assert_eq!(
            packed.utilization[&Category::RetrievedCode].budget,
            Some(40)
        );
    }

    #[test]
    fn assemble_groups_sections_and_orders_history() {
        let mut h2 = cand("history:2", Category::History, false, 0.0, "second");
        h2.message = Some(2);
        h2.role = Some(Role::Assistant);
        let mut h0 = cand("history:0", Category::History, false, 0.0, "first");
        h0.message = Some(0);
        h0.role = Some(Role::User);
        let task = cand("task", Category::Task, true, 0.0, "fix the bug");
        let packed = pack(
            vec![h2, task, h0],
            vec![],
            10_000,
            TaskKind::Feature,
            &EstimatingCounter,
        );
        let msgs = assemble_messages(&packed.selected);
        assert_eq!(msgs[0].role, Role::System);
        assert!(msgs[0].content.contains("## Task"));
        assert_eq!(msgs[1].content, "first");
        assert_eq!(msgs[2].content, "second");
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn render_adds_heading_only_when_present() {
        let mut c = cand("x", Category::RetrievedCode, false, 0.0, "body");
        assert_eq!(render(&c, "body"), "body");
        c.heading = "### a.rs".into();
        assert_eq!(render(&c, "body"), "### a.rs\nbody");
    }
}
