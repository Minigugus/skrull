use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::rc::Rc;
use alloc::vec::Vec;

use crate::lexer::{Token, TokenKind};
use crate::lexer::TokenKind::*;

pub type Error = Cow<'static, str>;
pub type Result<T> = core::result::Result<T, Error>;

trait MoreDetails<T> {
    fn when_parsing(self, node: &str) -> Result<T>;
    fn currently_at(self, tokens: &Vec<Token>) -> Result<T>;
}

impl<T> MoreDetails<T> for Result<T> {
    fn when_parsing(self, node: &str) -> Result<T> {
        self.map_err(|e| format!("while parsing {node}: {e}").into())
    }

    fn currently_at(self, tokens: &Vec<Token>) -> Result<T> {
        self.map_err(|e| format!("{e} (remaining tokens are {tokens:?})").into())
    }
}

trait Expected<T> {
    fn unwrap_or_expected(self, expected: impl Into<Error>) -> Result<T>;
    fn unwrap_or_expected_at(self, tokens: &mut Vec<Token>, expected: impl Into<Error>) -> Result<T>;
}

impl<T> Expected<T> for Option<T> {
    fn unwrap_or_expected(self, expected: impl Into<Error>) -> Result<T> {
        if let Some(value) = self {
            Ok(value)
        } else {
            Err(expected.into())
        }
    }

    fn unwrap_or_expected_at(self, _tokens: &mut Vec<Token>, expected: impl Into<Error>) -> Result<T> {
        self.unwrap_or_expected(expected)
    }
}

impl<T> Expected<T> for Result<Option<T>> {
    fn unwrap_or_expected(self, expected: impl Into<Error>) -> Result<T> {
        self?.unwrap_or_expected(expected)
    }

    fn unwrap_or_expected_at(self, tokens: &mut Vec<Token>, expected: impl Into<Error>) -> Result<T> {
        self?.unwrap_or_expected_at(tokens, expected)
    }
}

trait FromKeywordList: Sized {
    fn from_keyword_list(keywords: &mut Vec<Identifier>) -> Result<Self>;
}

impl FromKeywordList for Visibility {
    fn from_keyword_list(keywords: &mut Vec<Identifier>) -> Result<Self> {
        let visibility = match keywords.contains(&Identifier("pub")) {
            true => Visibility::Pub,
            false => Visibility::Default
        };
        keywords.retain(|k| k.0 != "pub");
        Ok(visibility)
    }
}

impl FromKeywordList for Mutability {
    fn from_keyword_list(keywords: &mut Vec<Identifier>) -> Result<Self> {
        let mutability = match keywords.contains(&Identifier("mut")) {
            true => Mutability::Mutable,
            false => Mutability::Default
        };
        keywords.retain(|k| k.0 != "pub");
        Ok(mutability)
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct BlockExpression<'a> {
    pub expressions: Vec<Expression<'a>>,
    pub remainder: Option<Rc<Expression<'a>>>,
}

#[derive(Eq, PartialEq, Debug)]
pub struct MatchExpression<'a> {
    pub expression: Rc<Expression<'a>>,
    pub cases: Vec<MatchCase<'a>>,
}

#[derive(Eq, PartialEq, Debug)]
pub struct MatchCase<'a> {
    pub pattern: Rc<MatchPattern<'a>>,
    pub body: Rc<Expression<'a>>,
    pub guard: Option<Rc<Expression<'a>>>,
}

#[derive(Eq, PartialEq, Debug)]
pub struct Qualifier<'a> {
    // Path { container: Identifier<'a>, left: Rc<Qualifier<'a>> },
    // Type(Identifier<'a>),
    pub parent: Option<Rc<Qualifier<'a>>>,
    pub segment: Identifier<'a>,
}

#[derive(Eq, PartialEq, Debug)]
pub enum MatchPattern<'a> {
    Unit,
    Wildcard,
    Variable(Identifier<'a>),
    Union(Vec<MatchPattern<'a>>),
    NumberLiteral(i64),
    StringLiteral(&'a str),
    IsEnumOrType(Qualifier<'a>),
    TupleStruct { typ: Qualifier<'a>, params: Vec<MatchPattern<'a>>, exact: bool },
    FieldStruct { typ: Qualifier<'a>, params: Vec<(Identifier<'a>, MatchPattern<'a>)>, exact: bool },
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Expression<'a> {
    Unit,
    Literal(i64),
    Identifier(Identifier<'a>),
    Block(Rc<BlockExpression<'a>>),
    Match(Rc<MatchExpression<'a>>),
    Neg(Rc<Expression<'a>>),
    Add(Rc<Expression<'a>>, Rc<Expression<'a>>),
    Mul(Rc<Expression<'a>>, Rc<Expression<'a>>),
    Gt(Rc<Expression<'a>>, Rc<Expression<'a>>),
    If(Rc<Expression<'a>>, Rc<BlockExpression<'a>>, Rc<BlockExpression<'a>>),
    Call(Identifier<'a>, Vec<Expression<'a>>),
    Create(Identifier<'a>, Vec<(Identifier<'a>, Expression<'a>)>),
}

#[derive(Eq, PartialEq, Debug)]
pub struct FunctionDeclaration<'a> {
    pub visibility: Visibility,
    pub name: Identifier<'a>,
    pub parameters: Vec<VariableSymbolDeclaration<'a>>,
    pub ret_type: Option<Type<'a>>,
    pub body: BlockExpression<'a>,
}

#[derive(Eq, PartialEq, Debug)]
pub struct VariableSymbolDeclaration<'a> {
    pub mutability: Mutability,
    pub name: Identifier<'a>,
    pub typ: Option<Type<'a>>,
}

#[derive(Eq, PartialEq, Debug)]
pub enum Mutability {
    Mutable,
    Default,
}

impl From<Mutability> for bool {
    fn from(value: Mutability) -> Self {
        matches!(value, Mutability::Mutable)
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum Visibility {
    Pub,
    Default,
}

#[derive(Eq, Copy, Clone, PartialEq, Debug)]
pub enum Type<'a> {
    I16,
    U32,
    I64,
    F64,
    Unit,
    Usize,
    Identifier(Identifier<'a>),
}

#[derive(Eq, Copy, Clone, PartialEq, Debug)]
pub struct Identifier<'a>(pub &'a str);

impl<'a> AsRef<str> for Identifier<'a> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct NamedField<'a> {
    pub visibility: Visibility,
    pub name: Identifier<'a>,
    pub typ: Type<'a>,
}

#[derive(Eq, PartialEq, Debug)]
pub struct TupleField<'a> {
    pub visibility: Visibility,
    pub typ: Type<'a>,
}

#[derive(Eq, PartialEq, Debug)]
pub enum Fields<'a> {
    NamedFields(Vec<NamedField<'a>>),
    TupleFields(Vec<TupleField<'a>>),
    Unit,
}

#[derive(Eq, PartialEq, Debug)]
pub struct EnumVariant<'a> {
    pub name: Identifier<'a>,
    pub fields: Fields<'a>,
}

#[derive(Eq, PartialEq, Debug)]
pub struct StructOrEnum<'a, T> {
    pub visibility: Visibility,
    pub name: Identifier<'a>,
    pub body: T,
}

pub type Struct<'a> = StructOrEnum<'a, Fields<'a>>;
pub type Enum<'a> = StructOrEnum<'a, Vec<EnumVariant<'a>>>;

fn peek_token<'a, 'b>(tokens: &'a mut Vec<Token<'b>>) -> Option<&'a TokenKind<'b>> {
    tokens.get(0).map(|t| &t.kind)
}

fn eat_token(tokens: &mut Vec<Token>, expected: TokenKind) -> Option<()> {
    match peek_token(tokens) {
        Some(token) if *token == expected => {
            tokens.remove(0);
            Some(())
        }
        _ => None
    }
}

fn parse_group<'a, T, F: Fn(&mut Vec<Token<'a>>) -> Result<T>>(
    tokens: &mut Vec<Token<'a>>,
    parse: F,
    open: TokenKind,
    close: TokenKind,
) -> Result<Vec<T>> {
    parse_optional_group_with_dotdot(tokens, parse, open, close)
        .unwrap_or_expected_at(tokens, "expected an opening token")
        .map(|(v, _)| v)
}

fn parse_optional_group_with_dotdot<'a, T, F: Fn(&mut Vec<Token<'a>>) -> Result<T>>(
    tokens: &mut Vec<Token<'a>>,
    parse: F,
    open: TokenKind,
    close: TokenKind
) -> Result<(Option<(Vec<T>, bool)>)> {
    if !eat_token(tokens, open).is_some() {
        return Ok(None);
    }

    let mut items = vec![];
    let mut dotdot_at_end = false;

    while !eat_token(tokens, close).is_some() {
        items.push(parse(tokens)?);
        if !eat_token(tokens, Comma).is_some() {
            dotdot_at_end = eat_token(tokens, DotDot).is_some();
            eat_token(tokens, close).unwrap_or_expected_at(tokens, "expected a comma or closing token")?;
            break;
        }
    }

    Ok(Some((items, dotdot_at_end)))
}

fn parse_visibility(tokens: &mut Vec<Token>) -> Visibility {
    if let Some(Symbol("pub")) = peek_token(tokens) {
        tokens.remove(0);
        Visibility::Pub
    } else {
        Visibility::Default
    }
}

fn parse_identifier<'a>(tokens: &mut Vec<Token<'a>>) -> Option<Identifier<'a>> {
    if let Some(Symbol(_)) = peek_token(tokens) {
        let Symbol(name) = tokens.remove(0).kind else { return None; };
        Some(Identifier(name))
    } else {
        None
    }
}

// fn parse_qualifier<'a>(tokens: &mut Vec<Token<'a>>) -> Result<Qualifier<'a>> {
//     let segment = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected an identifier")?;
//     Ok(if !eat_token(tokens, ColonColon).is_some() {
//         Qualifier::Type(segment)
//     } else {
//         Qualifier::Path { container: segment, left: Rc::new(parse_qualifier(tokens)?) }
//     })
// }

fn parse_qualifier_inner<'a>(segment: Identifier<'a>, parent: Option<Rc<Qualifier<'a>>>, tokens: &mut Vec<Token<'a>>) -> Result<Qualifier<'a>> {
    let qualifier = Qualifier {
        parent,
        segment,
    };
    Ok(if !eat_token(tokens, ColonColon).is_some() {
        qualifier
    } else {
        parse_qualifier_inner(
            parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected an identifier")?,
            Some(Rc::new(qualifier)),
            tokens
        )?
    })
}

fn parse_qualifier<'a>(segment: Identifier<'a>, tokens: &mut Vec<Token<'a>>) -> Result<Qualifier<'a>> {
    parse_qualifier_inner(
        segment,
        None,
        tokens
    )
}

fn parse_identifier_and_keywords<'a, T: FromKeywordList>(
    tokens: &mut Vec<Token<'a>>,
    until: TokenKind,
) -> Result<Option<(Identifier<'a>, T)>> {
    let Some(mut identifier) = parse_identifier(tokens) else { return Ok(None); };
    let mut keywords = vec![];
    while !eat_token(tokens, until.clone()).is_some() {
        keywords.push(identifier);
        if let Some(id) = parse_identifier(tokens) {
            identifier = id;
        } else {
            return Ok(None);
        }
    }
    Ok(Some((identifier, T::from_keyword_list(&mut keywords)?)))
}

fn parse_type<'a>(tokens: &mut Vec<Token<'a>>) -> Result<Option<Type<'a>>> {
    if eat_token(tokens, ParenthesisOpen).is_some() {
        eat_token(tokens, ParenthesisClose)
            .unwrap_or_expected_at(tokens, "expected a closing parenthesis for unit type")
            .when_parsing("the unit type")?;
        return Ok(Some(Type::Unit));
    }
    Ok(match peek_token(tokens) {
        Some(Symbol(_)) => Some(match parse_identifier(tokens)
            .unwrap_or_expected_at(tokens, "expected a type identifier")
            .when_parsing("the type identifier")? {
            Identifier("i16") => Type::I16,
            Identifier("u32") => Type::U32,
            Identifier("i64") => Type::I64,
            Identifier("f64") => Type::F64,
            Identifier("usize") => Type::Usize,
            name => Type::Identifier(name)
        }),
        _ => None,
    })
}

fn parse_named_field<'a>(tokens: &mut Vec<Token<'a>>) -> Result<NamedField<'a>> {
    // let visibility = parse_visibility(tokens);
    // let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a field name")?;
    let (name, visibility) = parse_identifier_and_keywords(tokens, Colon)
        .when_parsing("the named field identifier and visibility")?
        .unwrap_or_expected_at(tokens, "expected the `pub` keyword or an field identifier")?;
    let r#type = parse_type(tokens)
        .when_parsing("the named field type")?
        .unwrap_or_expected_at(tokens, "expected a field type")?;

    Ok(NamedField {
        visibility,
        name,
        typ: r#type,
    })
}

fn parse_tuple_field<'a>(tokens: &mut Vec<Token<'a>>) -> Result<TupleField<'a>> {
    let visibility = parse_visibility(tokens);
    // let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a field name")?;
    // let (name, visibility) = parse_identifier_and_keywords(tokens, Colon).unwrap_or_expected_at(tokens, "expected the `pub` keyword or an field identifier")?;
    let r#type = parse_type(tokens)
        .when_parsing("the tuple field type")?
        .unwrap_or_expected_at(tokens, "expected a field type")?;

    Ok(TupleField {
        visibility,
        typ: r#type,
    })
}

pub fn parse_fields<'a>(tokens: &mut Vec<Token<'a>>) -> Result<Fields<'a>> {
    Ok(match peek_token(tokens) {
        Some(BraceOpen) => Fields::NamedFields(parse_group(tokens, parse_named_field, BraceOpen, BraceClose)
            .when_parsing("a named field")?),
        Some(ParenthesisOpen) => Fields::TupleFields(parse_group(tokens, parse_tuple_field, ParenthesisOpen, ParenthesisClose)
            .when_parsing("a tuple field")?),
        _ => Fields::Unit
    })
}

fn parse_struct_inner<'a>(tokens: &mut Vec<Token<'a>>, visibility: Visibility) -> Result<Struct<'a>> {
    eat_token(tokens, Symbol("struct")).unwrap_or_expected_at(tokens, "expected the struct keyword")?;
    let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a struct name")?;
    let body = parse_fields(tokens)?;
    if !matches!(body, Fields::NamedFields(_)) {
        eat_token(tokens, Semicolon).unwrap_or_expected("expected `;` after struct declaration")?;
    }

    Ok(StructOrEnum {
        visibility,
        name,
        body,
    })
}

pub fn parse_struct<'a>(tokens: &mut Vec<Token<'a>>) -> Result<Struct<'a>> {
    let visibility = parse_visibility(tokens);

    parse_struct_inner(tokens, visibility)
}

pub fn parse_enum_variant<'a>(tokens: &mut Vec<Token<'a>>) -> Result<EnumVariant<'a>> {
    // let visibility = parse_visibility(tokens);
    // eat_token(tokens, Symbol("enum")).unwrap_or_expected_at(tokens, "expected the enum keyword")?;
    let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a enum name")?;
    let fields = parse_fields(tokens)?;

    Ok(EnumVariant {
        name,
        fields,
    })
}

fn parse_enum_inner<'a>(tokens: &mut Vec<Token<'a>>, visibility: Visibility) -> Result<Enum<'a>> {
    eat_token(tokens, Symbol("enum")).unwrap_or_expected_at(tokens, "expected the enum keyword")?;
    let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a enum name")?;
    let body = parse_group(tokens, parse_enum_variant, BraceOpen, BraceClose)
        .when_parsing("the enum body")?;

    Ok(StructOrEnum {
        visibility,
        name,
        body,
    })
}

pub fn parse_enum<'a>(tokens: &mut Vec<Token<'a>>) -> Result<Enum<'a>> {
    let visibility = parse_visibility(tokens);

    parse_enum_inner(tokens, visibility)
}

pub fn parse_field_init<'a>(tokens: &mut Vec<Token<'a>>) -> Result<(Identifier<'a>, Expression<'a>)> {
    let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a field name")?;
    let init = if eat_token(tokens, Colon).is_some() {
        parse_expression(false, tokens)?
    } else {
        Expression::Identifier(name.clone())
    };

    Ok((name, init))
}

pub fn parse_match_field_pattern<'a>(tokens: &mut Vec<Token<'a>>) -> Result<(Identifier<'a>, MatchPattern<'a>)> {
    let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a field name")?;
    let pattern = if eat_token(tokens, Colon).is_some() {
        parse_match_pattern(tokens)?
    } else {
        MatchPattern::Variable(name.clone())
    };

    Ok((name, pattern))
}

pub fn parse_match_pattern_without_union<'a>(tokens: &mut Vec<Token<'a>>) -> Result<MatchPattern<'a>> {
    if eat_token(tokens, Underscore).is_some() {
        return Ok(MatchPattern::Wildcard);
    }

    let typ = match tokens.remove(0).kind {
        Symbol(name) => parse_qualifier(Identifier(name), tokens)?,
        Number(n) => return Ok(MatchPattern::NumberLiteral(n)),
        _ => return Err("expected a match pattern")?
    };

    Ok(if let Some((params, exact)) = parse_optional_group_with_dotdot(tokens, parse_match_pattern, ParenthesisOpen, ParenthesisClose)? {
        MatchPattern::TupleStruct {
            typ,
            params,
            exact,
        }
    } else if let Some((params, exact)) = parse_optional_group_with_dotdot(tokens, parse_match_field_pattern, BraceOpen, BraceClose)? {
        MatchPattern::FieldStruct {
            typ,
            params,
            exact,
        }
    } else if typ.parent.is_none() {
        MatchPattern::Variable(typ.segment)
    } else {
        MatchPattern::IsEnumOrType(typ)
    })
}

pub fn parse_match_pattern<'a>(tokens: &mut Vec<Token<'a>>) -> Result<MatchPattern<'a>> {
    let mut prev = parse_match_pattern_without_union(tokens)?;
    if eat_token(tokens, Pipe).is_none() {
        return Ok(prev);
    }
    let mut patterns = vec![prev];
    loop {
        patterns.push(parse_match_pattern_without_union(tokens)?);
        if eat_token(tokens, Pipe).is_none() {
            break;
        }
    }
    return Ok(MatchPattern::Union(patterns));
}

pub fn parse_match_case<'a>(tokens: &mut Vec<Token<'a>>) -> Result<MatchCase<'a>> {
    let pattern = parse_match_pattern(tokens)?;

    let guard = if eat_token(tokens, Symbol("if")).is_some() {
        Some(Rc::new(parse_expression(false, tokens)?))
    } else {
        None
    };

    eat_token(tokens, DoubleArrow).unwrap_or_expected_at(tokens, "expected a double arrow `=>`")?;

    let body = parse_expression(false, tokens)?;

    Ok(MatchCase {
        pattern: Rc::new(pattern),
        body: Rc::new(body),
        guard,
    })
}

pub fn parse_low_expression<'a>(narrowed: bool, tokens: &mut Vec<Token<'a>>) -> Result<Expression<'a>> {
    if let Some(BraceOpen) = peek_token(tokens) {
        return Ok(Expression::Block(Rc::new(parse_block_expression(tokens)?)));
    }
    match tokens.remove(0).kind {
        Number(v) => Ok(Expression::Literal(v)),
        MinusSign => Ok(Expression::Neg(Rc::new(parse_low_expression(narrowed, tokens)?))),
        ParenthesisOpen => {
            Ok(if eat_token(tokens, ParenthesisClose).is_some() {
                Expression::Unit
            } else {
                let nested = parse_expression(false, tokens)
                    .when_parsing("a nested expression")?;
                eat_token(tokens, ParenthesisClose).unwrap_or_expected_at(tokens, "expected a closing parenthesis")?;
                nested
            })
        }
        Symbol("if") => {
            let cond = parse_expression(true, tokens)
                .when_parsing("a `if` condition")?;
            let on_true = parse_block_expression(tokens)
                .when_parsing("the `if` true branch")?;
            eat_token(tokens, Symbol("else")).unwrap_or_expected_at(tokens, "expected `else` keyword")?;
            let on_false = parse_block_expression(tokens)
                .when_parsing("the `if` false branch")?;
            Ok(Expression::If(
                Rc::new(cond),
                Rc::new(on_true),
                Rc::new(on_false),
            ))
        }
        Symbol("match") => {
            let expr = parse_expression(true, tokens)
                .when_parsing("a `match` expression")?;
            let cases = parse_group(tokens, parse_match_case, BraceOpen, BraceClose)
                .when_parsing("a `match` case")
                .currently_at(tokens)?;
            Ok(Expression::Match(Rc::new(MatchExpression {
                expression: Rc::new(expr),
                cases,
            })))
        }
        Symbol(n) => {
            let identifier = Identifier(n);
            if let Some(ParenthesisOpen) = peek_token(tokens) {
                let arguments = parse_group(tokens, |t| parse_expression(false, t), ParenthesisOpen, ParenthesisClose)
                    .when_parsing("function call arguments")?;
                return Ok(Expression::Call(identifier, arguments));
            }
            if !narrowed {
                if let Some(BraceOpen) = peek_token(tokens) {
                    let arguments = parse_group(tokens, |t| parse_field_init(t), BraceOpen, BraceClose)
                        .when_parsing("a struct initializer")?;
                    return Ok(Expression::Create(identifier, arguments));
                }
            }
            Ok(Expression::Identifier(identifier))
        }
        other => Result::currently_at(Err(format!("expected an expression but got {other:?}").into()), tokens)
    }
}

pub fn parse_middle_low_expression<'a>(narrowed: bool, tokens: &mut Vec<Token<'a>>) -> Result<Expression<'a>> {
    let mut prev = parse_low_expression(narrowed, tokens)?;
    loop {
        prev = if eat_token(tokens, StarSign).is_some() {
            Expression::Mul(Rc::new(prev), Rc::new(parse_low_expression(narrowed, tokens)?))
        } else {
            return Ok(prev);
        };
    }
}

pub fn parse_middle_expression<'a>(narrowed: bool, tokens: &mut Vec<Token<'a>>) -> Result<Expression<'a>> {
    let mut prev = parse_middle_low_expression(narrowed, tokens)?;
    loop {
        prev = if eat_token(tokens, PlusSign).is_some() {
            Expression::Add(Rc::new(prev), Rc::new(parse_middle_low_expression(narrowed, tokens)?))
        } else {
            return Ok(prev);
        };
    }
}

pub fn parse_middle_high_expression<'a>(narrowed: bool, tokens: &mut Vec<Token<'a>>) -> Result<Expression<'a>> {
    let mut prev = parse_middle_expression(narrowed, tokens)?;
    loop {
        prev = if eat_token(tokens, GreaterThan).is_some() {
            Expression::Gt(Rc::new(prev), Rc::new(parse_middle_expression(narrowed, tokens)?))
        } else {
            return Ok(prev);
        };
    }
}

pub fn parse_expression<'a>(narrowed: bool, tokens: &mut Vec<Token<'a>>) -> Result<Expression<'a>> {
    parse_middle_high_expression(narrowed, tokens)
}

pub fn parse_block_expression<'a>(tokens: &mut Vec<Token<'a>>) -> Result<BlockExpression<'a>> {
    let mut expressions = vec![];
    let mut remainder = None;

    eat_token(tokens, BraceOpen)
        .unwrap_or_expected_at(tokens, "expected an opening brace")
        .currently_at(tokens)?;
    while !eat_token(tokens, BraceClose).is_some() {
        eat_token(tokens, Semicolon)
            .map_or_else(|| Some(()), |_| None)
            .unwrap_or_expected_at(tokens, "extra semi-colon")
            .currently_at(tokens)?;

        let expr = parse_expression(false, tokens)
            .when_parsing("a block expression")?;
        if eat_token(tokens, Semicolon).is_some() {
            expressions.push(expr);
        } else {
            remainder = Some(Rc::new(expr));
            eat_token(tokens, BraceClose)
                .unwrap_or_expected_at(tokens, "expected a semi-colon or closing brace")
                .currently_at(tokens)?;
            break;
        }
    }

    Ok(BlockExpression {
        expressions,
        remainder,
    })
}

pub fn parse_parameter<'a>(tokens: &mut Vec<Token<'a>>) -> Result<VariableSymbolDeclaration<'a>> {
    let (name, mutability) = parse_identifier_and_keywords(tokens, Colon).unwrap_or_expected_at(tokens, "expected an identifier")?;
    let typ = parse_type(tokens)?;

    Ok(VariableSymbolDeclaration {
        mutability,
        name,
        typ,
    })
}

fn parse_return_type<'a>(tokens: &mut Vec<Token<'a>>) -> Result<Option<Type<'a>>> {
    Ok(if eat_token(tokens, SimpleArrow).is_some() {
        Some(parse_type(tokens).unwrap_or_expected_at(tokens, "expected a return type")?)
    } else {
        None
    })
}

fn parse_function_inner<'a>(tokens: &mut Vec<Token<'a>>, visibility: Visibility) -> Result<FunctionDeclaration<'a>> {
    eat_token(tokens, Symbol("fn")).unwrap_or_expected_at(tokens, "expected the `fn` keyword")?;
    let name = parse_identifier(tokens).unwrap_or_expected_at(tokens, "expected a function name")?;
    let parameters = parse_group(tokens, parse_parameter, ParenthesisOpen, ParenthesisClose)
        .when_parsing("function declaration parameters")?;
    let ret_type = parse_return_type(tokens)?;
    let body = parse_block_expression(tokens)?;

    Ok(FunctionDeclaration {
        visibility,
        name,
        parameters,
        ret_type,
        body,
    })
}

pub fn parse_function_declaration<'a>(tokens: &mut Vec<Token<'a>>) -> Result<FunctionDeclaration<'a>> {
    let visibility = parse_visibility(tokens);

    parse_function_inner(tokens, visibility)
}

pub enum Declaration<'a> {
    Enum(Enum<'a>),
    Function(FunctionDeclaration<'a>),
    Struct(Struct<'a>),
}

pub fn parse_declaration<'a>(tokens: &mut Vec<Token<'a>>) -> Result<Declaration<'a>> {
    let visibility = parse_visibility(tokens);

    Ok(match peek_token(tokens) {
        Some(Symbol("enum")) => Declaration::Enum(parse_enum_inner(tokens, visibility)?),
        Some(Symbol("fn")) => Declaration::Function(parse_function_inner(tokens, visibility)?),
        Some(Symbol("struct")) => Declaration::Struct(parse_struct_inner(tokens, visibility)?),
        kind => Err(format!("expected a struct, enum or function declaration, got {kind:?}"))?
    })
}


#[test]
fn it_tokenize_struct_with_keywords_as_identifiers() -> Result<()> {
    let mut tokens = Token::parse_ascii(r#"pub struct Token {
  struct: TokenKind,
  pub pub: usize,
}"#)?;

    let root = parse_struct(&mut tokens)?;

    assert_eq!(
        Struct {
            visibility: Visibility::Pub,
            name: Identifier("Token"),
            body: Fields::NamedFields(vec![
                NamedField {
                    visibility: Visibility::Default,
                    name: Identifier("struct"),
                    typ: Type::Identifier(Identifier("TokenKind")),
                },
                NamedField {
                    visibility: Visibility::Pub,
                    name: Identifier("pub"),
                    typ: Type::Usize,
                },
            ]),
        },
        root
    );

    Ok(())
}

#[test]
fn it_tokenize_enum() -> Result<()> {
    let mut tokens = /*language=rust*/Token::parse_ascii(r#"pub enum TokenKind {
    Equal,
    Unexpected { character: char }
}"#)?;

    let root = parse_enum(&mut tokens)?;

    assert_eq!(
        Enum {
            visibility: Visibility::Pub,
            name: Identifier("TokenKind"),
            body: vec![
                EnumVariant {
                    name: Identifier("Equal"),
                    fields: Fields::Unit,
                },
                EnumVariant {
                    name: Identifier("Unexpected"),
                    fields: Fields::NamedFields(vec![
                        NamedField {
                            visibility: Visibility::Default,
                            name: Identifier("character"),
                            typ: Type::Identifier(Identifier("char")),
                        },
                    ]),
                },
            ],
        },
        root
    );

    Ok(())
}

#[test]
fn it_tokenize_block_expression() -> Result<()> {
    let mut tokens = /*language=rust*/Token::parse_ascii(r#"{
        2 + --3 + 5
    }"#)?;

    let root = parse_block_expression(&mut tokens)?;

    assert_eq!(
        BlockExpression {
            expressions: vec![],
            remainder: Some(Rc::new(Expression::Add(Rc::new(
                Expression::Add(Rc::new(
                    Expression::Literal(2)
                ), Rc::new(
                    Expression::Neg(Rc::new(Expression::Neg(Rc::new(Expression::Literal(3)))))
                ))
            ), Rc::new(
                Expression::Literal(5)
            )))),
        },
        root
    );

    Ok(())
}

#[test]
fn it_tokenize_function_declaration() -> Result<()> {
    let mut tokens = /*language=rust*/Token::parse_ascii(r#"pub fn ten(mut a: u32) {
        2 + 3 + 5
    }"#)?;

    let root = parse_function_declaration(&mut tokens)?;

    assert_eq!(
        FunctionDeclaration {
            visibility: Visibility::Pub,
            name: Identifier("ten"),
            parameters: vec![VariableSymbolDeclaration {
                mutability: Mutability::Mutable,
                name: Identifier("a"),
                typ: Some(Type::U32),
            }],
            ret_type: None,
            body: BlockExpression {
                expressions: vec![],
                remainder: Some(Rc::new(Expression::Add(Rc::new(
                    Expression::Add(Rc::new(
                        Expression::Literal(2)
                    ), Rc::new(
                        Expression::Literal(3)
                    ))
                ), Rc::new(
                    Expression::Literal(5)
                )))),
            },
        },
        root
    );

    Ok(())
}
