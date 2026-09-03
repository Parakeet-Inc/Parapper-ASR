//! A small, sherpa-onnx compatible context graph for RNNT hotwords.
//!
//! The graph keeps the score of the currently matched prefix in its state.
//! `forward` therefore returns a score delta, rather than the absolute score.
//! This is important when the graph falls back to a shorter suffix after a
//! mismatch: the score of the abandoned prefix is rolled back.

use std::collections::HashMap;

use anyhow::{Result, bail};
use unicode_normalization::UnicodeNormalization;

/// User-facing hotword definition. `surface` is the text emitted in the
/// transcript, while `readings` are optional spoken forms used for matching.
/// `phrase_score` is a positive phrase-level likelihood multiplier. When it
/// is present, it overrides the graph's default multiplier for this entry;
/// the graph converts it to log space internally and applies it once when a
/// complete path is emitted.
#[derive(Debug, Clone, PartialEq)]
pub struct HotwordEntry {
    pub surface: String,
    pub readings: Vec<String>,
    pub phrase_score: Option<f32>,
}

impl HotwordEntry {
    /// Creates an entry with no explicit spoken reading or phrase boost.
    ///
    /// # Errors
    ///
    /// Returns an error when `surface` is empty or whitespace-only.
    pub fn new(surface: impl Into<String>) -> Result<Self> {
        let surface = surface.into();
        if surface.trim().is_empty() {
            bail!("hotword surface must not be empty");
        }
        Ok(Self {
            surface,
            readings: Vec::new(),
            phrase_score: None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotwordPathKind {
    Surface,
    Reading,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HotwordTokenPath {
    pub tokens: Vec<usize>,
    pub entry_id: usize,
    pub surface: String,
    pub kind: HotwordPathKind,
    pub phrase_score: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HotwordMatch {
    pub start_token: usize,
    pub end_token: usize,
    pub entry_id: usize,
    pub surface: String,
    pub kind: HotwordPathKind,
}

/// Normalizes a Japanese reading to hiragana after compatibility normalization.
/// Half-width katakana therefore follows the same path as full-width katakana.
#[must_use]
pub fn normalize_reading(reading: &str) -> String {
    reading
        .nfkc()
        .map(|character| match character {
            '\u{30a1}'..='\u{30f6}' => char::from_u32(character as u32 - 0x60).unwrap_or(character),
            _ => character,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Node {
    prefix_score: f32,
    terminal: bool,
    terminal_paths: Vec<usize>,
    terminal_phrase_score: Option<f32>,
    fail: usize,
    output: Option<usize>,
    children: Vec<(usize, usize)>,
    injectable_children: Vec<usize>,
    continuation_tokens: Vec<usize>,
}

impl Node {
    fn root() -> Self {
        Self {
            prefix_score: 0.0,
            terminal: false,
            terminal_paths: Vec::new(),
            terminal_phrase_score: None,
            fail: 0,
            output: None,
            children: Vec::new(),
            injectable_children: Vec::new(),
            continuation_tokens: Vec::new(),
        }
    }

    fn child(&self, token: usize) -> Option<usize> {
        self.children
            .iter()
            .find_map(|&(child_token, child)| (child_token == token).then_some(child))
    }
}

/// A trie/context graph used to add a fixed bonus to configured token words.
///
/// The state is an index into the graph.  States describe the longest suffix
/// of the emitted token sequence that is also a hotword prefix.  A completed
/// hotword returns to the root (the sherpa-onnx non-strict behaviour), so a
/// following word can start immediately.
#[derive(Debug, Clone)]
pub struct HotwordContextGraph {
    nodes: Vec<Node>,
    paths: Vec<HotwordTokenPath>,
}

impl HotwordContextGraph {
    /// Builds a graph from token-id words and a positive finite token score.
    ///
    /// Empty words are rejected because they would be terminal at the root and
    /// have no meaningful streaming transition. Duplicate words are harmless
    /// and are merged into the existing trie path.
    ///
    /// # Errors
    ///
    /// Returns an error when a phrase is empty or `token_score` is not finite
    /// and strictly positive.
    pub fn new(words: Vec<Vec<usize>>, token_score: f32) -> Result<Self> {
        if !token_score.is_finite() || token_score <= 0.0 {
            bail!("hotword token_score must be finite and greater than zero");
        }
        if words.iter().any(Vec::is_empty) {
            bail!("hotword words must not be empty");
        }

        let paths = words
            .into_iter()
            .enumerate()
            .map(|(entry_id, tokens)| HotwordTokenPath {
                tokens,
                entry_id,
                surface: String::new(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            })
            .collect();
        Self::from_token_paths(paths, token_score)
    }

    /// Builds a graph from tokenized surface and reading paths with legacy
    /// per-token prefix scoring. Optional `phrase_score` values are still
    /// interpreted as likelihood multipliers and converted to a terminal log
    /// adjustment.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid scores, empty paths, ambiguous surfaces,
    /// or terminal-prefix collisions.
    pub fn from_token_paths(paths: Vec<HotwordTokenPath>, token_score: f32) -> Result<Self> {
        if !token_score.is_finite() || token_score <= 0.0 {
            bail!("hotword token_score must be finite and greater than zero");
        }
        Self::build(paths, PrefixScoring::PerToken(token_score))
    }

    /// Builds a graph whose completed paths receive one phrase-level
    /// likelihood multiplier, independent of their token length.
    ///
    /// During search, `ln(phrase_multiplier)` is distributed over each path's
    /// tokens. Shared prefixes retain the strongest provisional score and are
    /// corrected at divergence or completion, so every completed path receives
    /// exactly one `ln(phrase_multiplier)` bonus.
    ///
    /// # Errors
    ///
    /// Returns an error when the multiplier is not finite and greater than
    /// zero, or when a token path violates the normal graph contract.
    pub fn from_token_paths_with_phrase_multiplier(
        paths: Vec<HotwordTokenPath>,
        phrase_multiplier: f32,
    ) -> Result<Self> {
        if !phrase_multiplier.is_finite() || phrase_multiplier <= 0.0 {
            bail!("hotword phrase_multiplier must be finite and greater than zero");
        }
        Self::build(paths, PrefixScoring::PhraseMultiplier(phrase_multiplier))
    }

    fn build(paths: Vec<HotwordTokenPath>, scoring: PrefixScoring) -> Result<Self> {
        if paths.iter().any(|path| path.tokens.is_empty()) {
            bail!("hotword words must not be empty");
        }
        for path in &paths {
            if !path.surface.is_empty() && path.surface.trim().is_empty() {
                bail!("hotword surface must not be empty");
            }
            PrefixScoring::validate_phrase_multiplier(path.phrase_score)?;
        }

        // A reading path must never silently select one of two homophones.
        // Identical registrations for the same surface are harmless.
        let mut path_surfaces = HashMap::<Vec<usize>, String>::new();
        for path in &paths {
            if let Some(previous) = path_surfaces.insert(path.tokens.clone(), path.surface.clone())
                && previous != path.surface
            {
                bail!(
                    "hotword token path matches multiple surfaces: {previous:?} and {:?}",
                    path.surface
                );
            }
        }
        // In non-strict mode a terminal prefix resets to root immediately,
        // making the longer phrase unreachable. Reject this ambiguous setup
        // instead of silently biasing only the shorter surface.
        for (short_index, short) in paths.iter().enumerate() {
            if short.surface.is_empty() {
                continue;
            }
            for (long_index, long) in paths.iter().enumerate() {
                if short_index == long_index || short.tokens.len() >= long.tokens.len() {
                    continue;
                }
                if long.tokens.starts_with(&short.tokens) {
                    bail!(
                        "hotword path {:?} is a terminal prefix of {:?}",
                        short.surface,
                        long.surface
                    );
                }
            }
        }

        let mut graph = Self {
            nodes: vec![Node::root()],
            paths,
        };
        for path_id in 0..graph.paths.len() {
            let path = &graph.paths[path_id];
            let mut state = graph.root();
            let path_length = path.tokens.len();
            for (token_index, &token) in path.tokens.iter().enumerate() {
                let prefix_score =
                    scoring.prefix_score(path.phrase_score, token_index + 1, path_length);
                if scoring.injects_path(path.phrase_score)
                    && !graph.nodes[state].injectable_children.contains(&token)
                {
                    graph.nodes[state].injectable_children.push(token);
                }
                let next = graph.nodes[state].child(token);
                state = if let Some(next) = next {
                    graph.nodes[next].prefix_score =
                        graph.nodes[next].prefix_score.max(prefix_score);
                    next
                } else {
                    let next = graph.nodes.len();
                    graph.nodes.push(Node {
                        prefix_score,
                        terminal: false,
                        terminal_paths: Vec::new(),
                        terminal_phrase_score: None,
                        fail: 0,
                        output: None,
                        children: Vec::new(),
                        injectable_children: Vec::new(),
                        continuation_tokens: Vec::new(),
                    });
                    graph.nodes[state].children.push((token, next));
                    next
                };
            }
            graph.nodes[state].terminal = true;
            graph.nodes[state].terminal_paths.push(path_id);
            if let Some(score) = scoring.terminal_adjustment(path.phrase_score) {
                graph.nodes[state].terminal_phrase_score = Some(
                    graph.nodes[state]
                        .terminal_phrase_score
                        .map_or(score, |current| current.max(score)),
                );
            }
        }
        graph.build_failure_links();
        Ok(graph)
    }

    /// Returns the root state (always zero).
    #[must_use]
    pub fn root(&self) -> usize {
        0
    }

    /// Advances one token and returns `(score_delta, next_state)`.
    ///
    /// A mismatch rolls back the score of the abandoned prefix and retains a
    /// matching suffix when one exists.  Completing any configured word is
    /// non-strict: the state is reset to root after applying that word's
    /// cumulative score.
    #[must_use]
    pub fn forward(&self, state: usize, token: usize) -> (f32, usize) {
        let current = &self.nodes[state];
        let mut candidate = state;
        while candidate != self.root() && self.nodes[candidate].child(token).is_none() {
            candidate = self.nodes[candidate].fail;
        }
        let next = self.nodes[candidate].child(token).unwrap_or(self.root());

        let next_node = &self.nodes[next];
        let delta = next_node.prefix_score - current.prefix_score;
        let matched = next_node.terminal.then_some(next).or(next_node.output);
        if let Some(matched) = matched {
            // Match sherpa-onnx's non-strict ASR hotword mode: preserve the
            // completed word's score, remove any longer unmatched prefix, and
            // return to root so the next phrase starts independently.
            let phrase_score = self.nodes[matched].terminal_phrase_score.unwrap_or(0.0);
            (
                delta + self.nodes[matched].prefix_score - next_node.prefix_score + phrase_score,
                self.root(),
            )
        } else {
            (delta, next)
        }
    }

    /// Cancels a partially matched prefix at an utterance boundary.
    #[must_use]
    pub fn finalize(&self, state: usize) -> f32 {
        -self.nodes[state].prefix_score
    }

    /// Returns tokens that can continue the prefix represented by `state`.
    #[must_use]
    pub fn continuation_tokens(&self, state: usize) -> &[usize] {
        &self.nodes[state].continuation_tokens
    }

    /// Iterates tokens on arcs directly leaving `state`.
    ///
    /// Unlike [`Self::continuation_tokens`], this excludes arcs reachable only
    /// through failure links or by restarting from the root. It is intended
    /// for fixed-width pre-Top-K context biasing, so neutral and suppressing
    /// phrase multipliers are excluded as well.
    pub fn direct_continuation_tokens(&self, state: usize) -> impl Iterator<Item = usize> + '_ {
        self.nodes[state].injectable_children.iter().copied()
    }

    /// Finds non-overlapping longest configured paths in a completed token
    /// sequence. This is deliberately a post-processing pass: beam state does
    /// not need to carry surface strings or match spans.
    #[must_use]
    pub fn find_matches(&self, token_ids: &[usize]) -> Vec<HotwordMatch> {
        let mut matches = Vec::new();
        let mut cursor = 0;
        while cursor < token_ids.len() {
            let best = self
                .paths
                .iter()
                .enumerate()
                .filter_map(|(path_id, path)| {
                    let end = cursor.checked_add(path.tokens.len())?;
                    (end <= token_ids.len()
                        && token_ids[cursor..end] == path.tokens
                        && !path.surface.is_empty())
                    .then_some((path_id, end))
                })
                .max_by_key(|&(path_id, end)| {
                    let kind_rank = u8::from(self.paths[path_id].kind == HotwordPathKind::Reading);
                    (end, kind_rank, path_id)
                });
            if let Some((path_id, end)) = best {
                let path = &self.paths[path_id];
                matches.push(HotwordMatch {
                    start_token: cursor,
                    end_token: end,
                    entry_id: path.entry_id,
                    surface: path.surface.clone(),
                    kind: path.kind,
                });
                cursor = end;
            } else {
                cursor += 1;
            }
        }
        matches
    }

    fn build_failure_links(&mut self) {
        let mut queue = std::collections::VecDeque::new();
        let root_children: Vec<usize> = self.nodes[self.root()]
            .children
            .iter()
            .map(|&(_, child)| child)
            .collect();
        for child in root_children {
            self.nodes[child].fail = self.root();
            queue.push_back(child);
        }

        while let Some(state) = queue.pop_front() {
            let children = self.nodes[state].children.clone();
            for (token, child) in children {
                let mut fallback = self.nodes[state].fail;
                while fallback != self.root() && self.nodes[fallback].child(token).is_none() {
                    fallback = self.nodes[fallback].fail;
                }
                self.nodes[child].fail = self.nodes[fallback].child(token).unwrap_or(self.root());
                let fail = self.nodes[child].fail;
                self.nodes[child].output = self.nodes[fail]
                    .terminal
                    .then_some(fail)
                    .or(self.nodes[fail].output);
                queue.push_back(child);
            }
        }

        // A token may continue the current prefix, a failure suffix, or a new
        // root phrase. Request all of them from compact Gather so acoustic
        // Top-K cannot hide a configured hotword branch.
        for state in 0..self.nodes.len() {
            let mut tokens = Vec::new();
            let mut cursor = state;
            loop {
                tokens.extend(self.nodes[cursor].injectable_children.iter().copied());
                if cursor == self.root() {
                    break;
                }
                cursor = self.nodes[cursor].fail;
            }
            tokens.sort_unstable();
            tokens.dedup();
            self.nodes[state].continuation_tokens = tokens;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PrefixScoring {
    PerToken(f32),
    PhraseMultiplier(f32),
}

impl PrefixScoring {
    fn validate_phrase_multiplier(phrase_multiplier: Option<f32>) -> Result<()> {
        if phrase_multiplier.is_some_and(|multiplier| !multiplier.is_finite() || multiplier <= 0.0)
        {
            bail!("hotword phrase_multiplier must be finite and greater than zero");
        }
        Ok(())
    }

    fn effective_multiplier(self, phrase_multiplier: Option<f32>) -> f32 {
        match self {
            Self::PerToken(_) => phrase_multiplier.unwrap_or(1.0),
            Self::PhraseMultiplier(default) => phrase_multiplier.unwrap_or(default),
        }
    }

    fn injects_path(self, phrase_multiplier: Option<f32>) -> bool {
        match self {
            Self::PerToken(token_score) => token_score > 0.0,
            Self::PhraseMultiplier(_) => self.effective_multiplier(phrase_multiplier) > 1.0,
        }
    }

    fn terminal_adjustment(self, phrase_multiplier: Option<f32>) -> Option<f32> {
        match self {
            Self::PerToken(_) => phrase_multiplier.map(f32::ln),
            Self::PhraseMultiplier(_) => None,
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "hotword token counts are small and scores use the model's f32 domain"
    )]
    fn prefix_score(
        self,
        phrase_multiplier: Option<f32>,
        prefix_length: usize,
        path_length: usize,
    ) -> f32 {
        match self {
            Self::PerToken(token_score) => token_score * prefix_length as f32,
            Self::PhraseMultiplier(_) => {
                self.effective_multiplier(phrase_multiplier).ln() * prefix_length as f32
                    / path_length as f32
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HotwordContextGraph, HotwordEntry, HotwordPathKind, HotwordTokenPath};

    #[test]
    fn reading_normalization_accepts_full_and_half_width_katakana() {
        assert_eq!(super::normalize_reading("ﾊﾟﾗｸﾞﾗﾌ"), "ぱらぐらふ");
        assert_eq!(super::normalize_reading("パラグラフ"), "ぱらぐらふ");
        assert_eq!(super::normalize_reading("ぱらぐらふ"), "ぱらぐらふ");
    }

    #[test]
    fn entry_graph_applies_phrase_multiplier_only_on_completion() {
        let graph = HotwordContextGraph::from_token_paths(
            vec![HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 0,
                surface: "表記".to_string(),
                kind: HotwordPathKind::Reading,
                phrase_score: Some(2.0),
            }],
            0.5,
        )
        .unwrap();
        let (first, state) = graph.forward(graph.root(), 1);
        assert!((first - 0.5).abs() < 1.0e-6);
        let (last, state) = graph.forward(state, 2);
        assert!((last - (0.5 + 2.0_f32.ln())).abs() < 1.0e-6);
        assert_eq!(state, graph.root());
    }

    #[test]
    fn phrase_multiplier_defaults_to_x100_and_entry_value_overrides_it() {
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![
                HotwordTokenPath {
                    tokens: vec![1],
                    entry_id: 0,
                    surface: "default".to_string(),
                    kind: HotwordPathKind::Surface,
                    phrase_score: None,
                },
                HotwordTokenPath {
                    tokens: vec![2, 3],
                    entry_id: 1,
                    surface: "override".to_string(),
                    kind: HotwordPathKind::Surface,
                    phrase_score: Some(0.5),
                },
            ],
            100.0,
        )
        .expect("positive phrase multipliers should build");

        let (default_delta, state) = graph.forward(graph.root(), 1);
        assert_eq!(state, graph.root());
        assert!((default_delta - 100.0_f32.ln()).abs() < 1.0e-6);

        let (first, state) = graph.forward(graph.root(), 2);
        let (last, state) = graph.forward(state, 3);
        assert_eq!(state, graph.root());
        assert!(
            ((first + last) - 0.5_f32.ln()).abs() < 1.0e-6,
            "an entry multiplier overrides rather than multiplies the default x100"
        );
    }

    #[test]
    fn phrase_multiplier_prefixes_and_injection_follow_each_effective_entry_multiplier() {
        let path =
            |tokens: Vec<usize>, surface: &str, phrase_score: Option<f32>| HotwordTokenPath {
                tokens,
                entry_id: 0,
                surface: surface.to_string(),
                kind: HotwordPathKind::Surface,
                phrase_score,
            };
        let neutral = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![path(vec![1, 2], "neutral", Some(1.0))],
            100.0,
        )
        .expect("x1 should build");
        let (neutral_first, neutral_state) = neutral.forward(neutral.root(), 1);
        assert!(neutral_first.abs() < 1.0e-6);
        assert!(neutral.continuation_tokens(neutral_state).is_empty());

        let suppressed = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![path(vec![3, 4], "suppressed", Some(0.5))],
            100.0,
        )
        .expect("x0.5 should build");
        let (suppressed_first, suppressed_state) = suppressed.forward(suppressed.root(), 3);
        assert!(suppressed_first < 0.0);
        assert!(suppressed.continuation_tokens(suppressed_state).is_empty());

        let boosted = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![path(vec![5, 6], "boosted", None)],
            100.0,
        )
        .expect("default x100 should build");
        assert!(boosted.continuation_tokens(boosted.root()).contains(&5));
        let (boosted_first, boosted_state) = boosted.forward(boosted.root(), 5);
        assert!(boosted_first > 0.0);
        assert!(boosted.continuation_tokens(boosted_state).contains(&6));
    }

    #[test]
    fn shared_prefix_divergence_rolls_back_to_each_phrase_multiplier_exactly() {
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![
                HotwordTokenPath {
                    tokens: vec![1, 2],
                    entry_id: 0,
                    surface: "boosted".to_string(),
                    kind: HotwordPathKind::Surface,
                    phrase_score: None,
                },
                HotwordTokenPath {
                    tokens: vec![1, 3],
                    entry_id: 1,
                    surface: "suppressed".to_string(),
                    kind: HotwordPathKind::Surface,
                    phrase_score: Some(0.5),
                },
            ],
            100.0,
        )
        .expect("shared prefixes should build");

        let (shared, state) = graph.forward(graph.root(), 1);
        let (low_branch, state) = graph.forward(state, 3);
        assert_eq!(state, graph.root());
        assert!(
            ((shared + low_branch) - 0.5_f32.ln()).abs() < 1.0e-6,
            "taking the low-multiplier branch must rollback the shared x100 prefix"
        );

        let (shared, state) = graph.forward(graph.root(), 1);
        let (high_branch, state) = graph.forward(state, 2);
        assert_eq!(state, graph.root());
        assert!((shared + high_branch - 100.0_f32.ln()).abs() < 1.0e-6);
    }

    #[test]
    fn reading_match_returns_surface_and_token_span() {
        let graph = HotwordContextGraph::from_token_paths(
            vec![HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 7,
                surface: "斎藤".to_string(),
                kind: HotwordPathKind::Reading,
                phrase_score: None,
            }],
            0.5,
        )
        .unwrap();
        let matches = graph.find_matches(&[9, 1, 2, 8]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start_token, 1);
        assert_eq!(matches[0].end_token, 3);
        assert_eq!(matches[0].surface, "斎藤");
    }

    #[test]
    fn duplicate_reading_for_different_surfaces_is_rejected() {
        let paths = vec![
            HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 0,
                surface: "橋".to_string(),
                kind: HotwordPathKind::Reading,
                phrase_score: None,
            },
            HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 1,
                surface: "箸".to_string(),
                kind: HotwordPathKind::Reading,
                phrase_score: None,
            },
        ];
        assert!(HotwordContextGraph::from_token_paths(paths, 0.5).is_err());
    }

    #[test]
    fn duplicate_surface_and_reading_path_for_different_surfaces_is_rejected() {
        let paths = vec![
            HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 0,
                surface: "さいとう".to_string(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            },
            HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 1,
                surface: "斎藤".to_string(),
                kind: HotwordPathKind::Reading,
                phrase_score: None,
            },
        ];
        assert!(HotwordContextGraph::from_token_paths(paths, 0.5).is_err());
    }

    #[test]
    fn terminal_prefix_collision_is_rejected() {
        let paths = vec![
            HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 0,
                surface: "東京".to_string(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            },
            HotwordTokenPath {
                tokens: vec![1, 2, 3],
                entry_id: 1,
                surface: "東京都".to_string(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            },
        ];
        assert!(HotwordContextGraph::from_token_paths(paths, 0.5).is_err());
    }

    #[test]
    fn entry_constructor_rejects_empty_surface() {
        assert!(HotwordEntry::new("").is_err());
    }

    #[test]
    fn rejects_empty_and_invalid_scores() {
        assert!(HotwordContextGraph::new(vec![vec![]], 1.0).is_err());
        assert!(HotwordContextGraph::new(vec![vec![1]], 0.0).is_err());
        assert!(HotwordContextGraph::new(vec![vec![1]], -1.0).is_err());
        assert!(HotwordContextGraph::new(vec![vec![1]], f32::NAN).is_err());
        assert!(HotwordContextGraph::new(vec![vec![1]], f32::INFINITY).is_err());
    }

    #[test]
    fn duplicate_words_and_common_prefixes_share_states() {
        let graph =
            HotwordContextGraph::new(vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 4]], 0.5)
                .unwrap();
        let (first, state) = graph.forward(graph.root(), 1);
        assert!((first - 0.5).abs() < 1.0e-6);
        assert_eq!(graph.continuation_tokens(state), &[1, 2]);
        let (_, state) = graph.forward(state, 2);
        assert_eq!(graph.continuation_tokens(state), &[1, 3, 4]);
    }

    #[test]
    fn exact_match_scores_one_token_score_per_token() {
        let graph = HotwordContextGraph::new(vec![vec![1, 2, 3]], 0.7).unwrap();
        let mut state = graph.root();
        let mut total = 0.0;
        for token in [1, 2, 3] {
            let (delta, next) = graph.forward(state, token);
            total += delta;
            state = next;
        }
        assert!((total - 2.1).abs() < 1e-6);
        assert_eq!(state, graph.root());
    }

    #[test]
    fn phrase_multiplier_scores_completed_words_once_regardless_of_token_length() {
        let total_score = 2.0_f32;
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![
                HotwordTokenPath {
                    tokens: vec![1, 2],
                    entry_id: 0,
                    surface: "短語".to_string(),
                    kind: HotwordPathKind::Surface,
                    phrase_score: None,
                },
                HotwordTokenPath {
                    tokens: vec![1, 3, 4, 5],
                    entry_id: 1,
                    surface: "長い語".to_string(),
                    kind: HotwordPathKind::Surface,
                    phrase_score: None,
                },
            ],
            total_score.exp(),
        )
        .unwrap();

        for tokens in [[1, 2].as_slice(), [1, 3, 4, 5].as_slice()] {
            let mut state = graph.root();
            let mut accumulated = 0.0;
            for &token in tokens {
                let (delta, next) = graph.forward(state, token);
                accumulated += delta;
                state = next;
            }
            assert!((accumulated - total_score).abs() < 1.0e-6);
            assert_eq!(state, graph.root());
        }
    }

    #[test]
    fn phrase_multiplier_rolls_back_shared_prefix_and_supports_suppression() {
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![HotwordTokenPath {
                tokens: vec![1, 2, 3],
                entry_id: 0,
                surface: "抑制語".to_string(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            }],
            0.5,
        )
        .unwrap();
        assert!(graph.continuation_tokens(graph.root()).is_empty());
        let (prefix, state) = graph.forward(graph.root(), 1);
        assert!(prefix < 0.0);
        let (rollback, state) = graph.forward(state, 9);
        assert!((prefix + rollback).abs() < 1.0e-6);
        assert_eq!(state, graph.root());

        let mut state = graph.root();
        let mut accumulated = 0.0;
        for token in [1, 2, 3] {
            let (delta, next) = graph.forward(state, token);
            accumulated += delta;
            state = next;
        }
        assert!((accumulated - 0.5_f32.ln()).abs() < 1.0e-6);
        assert_eq!(state, graph.root());
    }

    #[test]
    fn neutral_phrase_multiplier_does_not_inject_acoustic_candidates() {
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 0,
                surface: "中立語".to_string(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            }],
            1.0,
        )
        .unwrap();
        assert!(graph.continuation_tokens(graph.root()).is_empty());
        let (first, state) = graph.forward(graph.root(), 1);
        let (second, state) = graph.forward(state, 2);
        assert!((first + second).abs() < 1.0e-6);
        assert_eq!(state, graph.root());
    }

    #[test]
    fn unfinished_prefix_is_cancelled_at_finalize() {
        let graph = HotwordContextGraph::new(vec![vec![1, 2, 3]], 0.7).unwrap();
        let (first, state) = graph.forward(graph.root(), 1);
        let (second, state) = graph.forward(state, 2);
        assert!((first + second + graph.finalize(state)).abs() < 1e-6);
    }

    #[test]
    fn mismatch_rolls_back_and_can_start_another_hotword() {
        let graph = HotwordContextGraph::new(vec![vec![1, 3], vec![2, 3]], 1.0).unwrap();
        let (_, state) = graph.forward(graph.root(), 1);
        let (rollback, state) = graph.forward(state, 2);
        assert!(rollback.abs() < 1.0e-6);
        assert_eq!(graph.continuation_tokens(state), &[1, 2, 3]);
        let (complete, state) = graph.forward(state, 3);
        assert!((complete - 1.0).abs() < 1.0e-6);
        assert_eq!(state, graph.root());
    }

    #[test]
    fn direct_continuations_exclude_root_and_failure_fallback_branches() {
        let graph = HotwordContextGraph::new(vec![vec![1, 3], vec![2, 3]], 1.0).unwrap();
        let (_, state) = graph.forward(graph.root(), 1);

        assert_eq!(
            graph.direct_continuation_tokens(state).collect::<Vec<_>>(),
            [3]
        );
        assert_eq!(graph.continuation_tokens(state), &[1, 2, 3]);
    }

    #[test]
    fn abab_overlap_completes_and_restarts_from_root() {
        let graph = HotwordContextGraph::new(vec![vec![1, 2, 1, 2]], 0.25).unwrap();
        let mut state = graph.root();
        let mut total = 0.0;
        for token in [1, 2, 1, 2] {
            let (delta, next) = graph.forward(state, token);
            total += delta;
            state = next;
        }
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(state, graph.root());
        let (delta, state) = graph.forward(state, 1);
        assert!((delta - 0.25).abs() < 1e-6);
        assert_ne!(state, graph.root());
    }
}
