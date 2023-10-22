use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum TokenKind<'a> {
    BraceOpen,
    BraceClose,
    // BracketOpen,
    // BracketClose,
    Colon,
    ColonColon,
    Comma,
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
    offset: usize,
}

impl<'a> Debug for Token<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Token { kind, offset } = self;
        write!(f, "{kind:?} @ {offset}")
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
                    offset,
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
    expected: &[(TokenKind<'static>, usize)],
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
            (Symbol("p"), 0),
            (Unexpected('😀'), 1),
            (Symbol("ub"), 5),
            (Unexpected('😁'), 8),
            (Semicolon, 12)
        ],
    )
}

#[test]
fn it_tokenize_mod() {
    use crate::lexer::TokenKind::*;

    assert_tokenize(
        "pub  mod   parser  ;",
        &[
            (Symbol("pub"), 0),
            (Symbol("mod"), 5),
            (Symbol("parser"), 11),
            (Semicolon, 19),
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
            (Symbol("pub"), 0),
            (Symbol("enum"), 4),
            (Symbol("Price"), 9),
            (BraceOpen, 15),
            (Symbol("Limit"), 19),
            (Comma, 24),
            (Symbol("Market"), 28),
            (Comma, 34),
            (Symbol("StopLimit"), 38),
            (BraceOpen, 48),
            (Symbol("stop_price"), 50),
            (Colon, 60),
            (Symbol("f64"), 62),
            (Comma, 65),
            (BraceClose, 67),
            (Comma, 68),
            (BraceClose, 70),
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
            (Symbol("pub"), 0),
            (Symbol("struct"), 4),
            (Symbol("Token"), 11),
            (BraceOpen, 17),
            (Symbol("kind"), 21),
            (Colon, 25),
            (Symbol("Token"), 27),
            (Comma, 32),
            (Symbol("offset"), 36),
            (Colon, 42),
            (Symbol("usize"), 44),
            (Comma, 49),
            (BraceClose, 51),
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
            (Symbol("fn"), 0),
            (Symbol("is_priced_type"), 3),
            (ParenthesisOpen, 17),
            (Symbol("type"), 18),
            (Colon, 22),
            (Symbol("Price"), 24),
            (ParenthesisClose, 29),
            (BraceOpen, 31),
            (Symbol("match"), 35),
            (Symbol("type"), 41),
            (BraceOpen, 46),
            (Symbol("Price"), 52),
            (ColonColon, 57),
            (Symbol("Limit"), 59),
            (Pipe, 65),
            (Symbol("Price"), 67),
            (ColonColon, 72),
            (Symbol("StopLimit"), 74),
            (BraceOpen, 84),
            (DotDot, 86),
            (BraceClose, 89),
            (DoubleArrow, 91),
            (Symbol("true"), 94),
            (Comma, 98),
            (Underscore, 104),
            (DoubleArrow, 106),
            (Symbol("false"), 109),
            (BraceClose, 117),
            (BraceClose, 119)
        ],
    )
}
