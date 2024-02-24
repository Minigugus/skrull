use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use core::ops::Range;

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum TokenKind<'a> {
    BraceOpen,
    BraceClose,
    // BracketOpen,
    // BracketClose,
    Colon,
    ColonColon,
    Comma,
    DocComment(&'a str),
    DoubleArrow,
    Equal,
    EqualEqual,
    GreaterOrEqual,
    GreaterThan,
    LowerOrEqual,
    LowerThan,
    Dot,
    DotDot,
    MinusSign,
    ParenthesisOpen,
    ParenthesisClose,
    Pipe,
    PlusSign,
    StarSign,
    Semicolon,
    SimpleArrow,
    Number(i64),
    Symbol(&'a str),
    Underscore,
    Unexpected(char),
}

#[derive(Eq, PartialEq, Clone)]
pub struct Token<'a> {
    pub kind: TokenKind<'a>,
    offset: Range<usize>,
}

impl<'a> Debug for Token<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Token { kind, offset } = self;
        write!(f, "{kind:?} @ {offset:?}")
    }
}

impl<'a> Token<'a> {
    pub fn parse_ascii(input: &'a str) -> Result<Vec<Self>, &'static str> {
        use TokenKind::*;

        let mut tokens = vec![];
        let mut offset = 0;
        let mut left = input;
        while !left.is_empty() {
            let mut chars = left.chars().peekable();
            let c = match chars.next() {
                None => break,
                Some(c) => c
            };
            let (consumed, kind) = match c {
                // whitespace
                c if char::is_ascii_whitespace(&c) => Self::skip_while(left, |c| c.is_whitespace()),

                // comment
                '/' if matches!(left.get(0..3), Some("///")) => Self::doc_comment(left),
                '/' if matches!(left.get(0..2), Some("//")) => Self::skip_while(left, |c| c != '\n'),

                // symbol
                c if char::is_ascii_digit(&c) => Self::number(&mut left)?,
                c if char::is_ascii_alphabetic(&c) => Self::symbol(&mut left),

                // complex punctuation
                // '=' => match left.get(0..2) {
                //     Some("=>") => Self::double_punct(Arrow),
                //     _ => Self::punct(Equal)
                // },
                '=' if matches!(left.get(0..2), Some("=>")) => Self::double_punct(DoubleArrow),
                '=' if matches!(left.get(0..2), Some("==")) => Self::double_punct(EqualEqual),
                '-' if matches!(left.get(0..2), Some("->")) => Self::double_punct(SimpleArrow),
                '>' if matches!(left.get(0..2), Some(">=")) => Self::double_punct(GreaterOrEqual),
                '<' if matches!(left.get(0..2), Some("<=")) => Self::double_punct(LowerOrEqual),

                // doubled punctuation
                ':' if matches!(left.get(0..2), Some("::")) => Self::double_punct(ColonColon),
                '.' if matches!(left.get(0..2), Some("..")) => Self::double_punct(DotDot),

                // simple punctuation
                '.' => Self::punct(Dot),
                '|' => Self::punct(Pipe),
                ',' => Self::punct(Comma),
                ':' => Self::punct(Colon),
                ';' => Self::punct(Semicolon),
                '{' => Self::punct(BraceOpen),
                '}' => Self::punct(BraceClose),
                '_' => Self::punct(Underscore),
                '(' => Self::punct(ParenthesisOpen),
                ')' => Self::punct(ParenthesisClose),
                '>' => Self::punct(GreaterThan),
                '<' => Self::punct(LowerThan),
                '=' => Self::punct(Equal),
                '+' => Self::punct(PlusSign),
                '*' => Self::punct(StarSign),
                '-' => Self::punct(MinusSign),

                // unmatched
                c => (c.len_utf8(), Some(Unexpected(c)))
                // _ => return Err("unexpected token")
            };
            if let Some(kind) = kind {
                tokens.push(Token {
                    kind,
                    offset: offset..(offset + consumed),
                });
            }
            left = &left[consumed..];
            offset += consumed;
        }
        Ok(tokens)
    }

    fn get_while<F: Fn(char) -> bool>(left: &str, criteria: F) -> usize {
        left
            .find(|c| !criteria(c))
            .unwrap_or(left.len())
    }

    fn punct(kind: TokenKind) -> (usize, Option<TokenKind>) {
        (1, Some(kind))
    }

    fn double_punct(kind: TokenKind) -> (usize, Option<TokenKind>) {
        (2, Some(kind))
    }

    fn skip_while<F: Fn(char) -> bool>(left: &str, criteria: F) -> (usize, Option<TokenKind>) {
        (Self::get_while(left, criteria), None)
    }

    fn number(left: &str) -> Result<(usize, Option<TokenKind>), &'static str> {
        let index = Self::get_while(
            left,
            |c| char::is_ascii_digit(&c),
        );

        Ok((index, Some(TokenKind::Number((&left[0..index]).parse().map_err(|_| "malformed number literal")?))))
    }

    fn doc_comment(left: &str) -> (usize, Option<TokenKind>) {
        let index = Self::get_while(
            left,
            |c| c != '\n',
        );

        (index, Some(TokenKind::DocComment(&left[3..index])))
    }

    fn symbol(left: &str) -> (usize, Option<TokenKind>) {
        let index = Self::get_while(
            left,
            |c| c == '_' || char::is_alphanumeric(c),
        );

        (index, Some(TokenKind::Symbol(&left[0..index])))
    }
}

fn assert_tokenize(
    input: &str,
    expected: &[(TokenKind<'static>, Range<usize>)],
) {
    let tokens = match Token::parse_ascii(input) {
        Ok(tokens) => tokens,
        Err(err) => panic!("couldn't parse tokens: {err}")
    };

    assert_eq!(
        tokens
            .into_iter()
            .map(|Token { kind, offset }| (kind, offset))
            .collect::<Vec<_>>()
            .as_slice(),
        expected
    );
}

#[test]
fn it_can_deal_with_utf8_characters() {
    use crate::lexer::TokenKind::*;

    assert_tokenize(
        "p😀ub 😁;",
        &[
            (Symbol("p"), 0..1),
            (Unexpected('😀'), 1..5),
            (Symbol("ub"), 5..7),
            (Unexpected('😁'), 8..12),
            (Semicolon, 12..13)
        ],
    )
}

#[test]
fn it_tokenize_mod() {
    use crate::lexer::TokenKind::*;

    assert_tokenize(
        "pub  mod   parser  ;",
        &[
            (Symbol("pub"), 0..3),
            (Symbol("mod"), 5..8),
            (Symbol("parser"), 11..17),
            (Semicolon, 19..20)
        ],
    )
}

#[test]
fn it_tokenize_enum() {
    use crate::lexer::TokenKind::*;

    assert_tokenize(
        r#"pub enum Price {
  Limit,
  Market,
  StopLimit { stop_price: f64, },
}"#,
        &[
            (Symbol("pub"), 0..3),
            (Symbol("enum"), 4..8),
            (Symbol("Price"), 9..14),
            (BraceOpen, 15..16),
            (Symbol("Limit"), 19..24),
            (Comma, 24..25),
            (Symbol("Market"), 28..34),
            (Comma, 34..35),
            (Symbol("StopLimit"), 38..47),
            (BraceOpen, 48..49),
            (Symbol("stop_price"), 50..60),
            (Colon, 60..61),
            (Symbol("f64"), 62..65),
            (Comma, 65..66),
            (BraceClose, 67..68),
            (Comma, 68..69),
            (BraceClose, 70..71)
        ],
    )
}

#[test]
fn it_tokenize_struct() {
    use crate::lexer::TokenKind::*;

    assert_tokenize(
        r#"pub struct Token {
  kind: Token,
  offset: usize,
}"#,
        &[
            (Symbol("pub"), 0..3),
            (Symbol("struct"), 4..10),
            (Symbol("Token"), 11..16),
            (BraceOpen, 17..18),
            (Symbol("kind"), 21..25),
            (Colon, 25..26),
            (Symbol("Token"), 27..32),
            (Comma, 32..33),
            (Symbol("offset"), 36..42),
            (Colon, 42..43),
            (Symbol("usize"), 44..49),
            (Comma, 49..50),
            (BraceClose, 51..52)
        ],
    )
}

#[test]
fn it_tokenize_fn() {
    use crate::lexer::TokenKind::*;

    assert_tokenize(
        r#"fn is_priced_type(type: Price) {
  match type {
    Price::Limit | Price::StopLimit { .. } => true,
    _ => false
  }
}"#,
        &[
            (Symbol("fn"), 0..2),
            (Symbol("is_priced_type"), 3..17),
            (ParenthesisOpen, 17..18),
            (Symbol("type"), 18..22),
            (Colon, 22..23),
            (Symbol("Price"), 24..29),
            (ParenthesisClose, 29..30),
            (BraceOpen, 31..32),
            (Symbol("match"), 35..40),
            (Symbol("type"), 41..45),
            (BraceOpen, 46..47),
            (Symbol("Price"), 52..57),
            (ColonColon, 57..59),
            (Symbol("Limit"), 59..64),
            (Pipe, 65..66),
            (Symbol("Price"), 67..72),
            (ColonColon, 72..74),
            (Symbol("StopLimit"), 74..83),
            (BraceOpen, 84..85),
            (DotDot, 86..88),
            (BraceClose, 89..90),
            (DoubleArrow, 91..93),
            (Symbol("true"), 94..98),
            (Comma, 98..99),
            (Underscore, 104..105),
            (DoubleArrow, 106..108),
            (Symbol("false"), 109..114),
            (BraceClose, 117..118),
            (BraceClose, 119..120)
        ],
    )
}

#[test]
fn it_tokenize_fn_with_doc_comments() {
    use crate::lexer::TokenKind::*;

    assert_tokenize(
        r#"
/// Whether or not this type of price accept a price
/// (e.g MARKET does not but LIMIT does)
fn is_priced_type(type: Price) {
  match type {
    Price::Limit | Price::StopLimit { .. } => true,
    _ => false
  }
}"#,
        &[
            (DocComment(" Whether or not this type of price accept a price"), 1..53),
            (DocComment(" (e.g MARKET does not but LIMIT does)"), 54..94),
            (Symbol("fn"), 95..97),
            (Symbol("is_priced_type"), 98..112),
            (ParenthesisOpen, 112..113),
            (Symbol("type"), 113..117),
            (Colon, 117..118),
            (Symbol("Price"), 119..124),
            (ParenthesisClose, 124..125),
            (BraceOpen, 126..127),
            (Symbol("match"), 130..135),
            (Symbol("type"), 136..140),
            (BraceOpen, 141..142),
            (Symbol("Price"), 147..152),
            (ColonColon, 152..154),
            (Symbol("Limit"), 154..159),
            (Pipe, 160..161),
            (Symbol("Price"), 162..167),
            (ColonColon, 167..169),
            (Symbol("StopLimit"), 169..178),
            (BraceOpen, 179..180),
            (DotDot, 181..183),
            (BraceClose, 184..185),
            (DoubleArrow, 186..188),
            (Symbol("true"), 189..193),
            (Comma, 193..194),
            (Underscore, 199..200),
            (DoubleArrow, 201..203),
            (Symbol("false"), 204..209),
            (BraceClose, 212..213),
            (BraceClose, 214..215)
        ],
    )
}
