use std::fmt;
use std::io::{self, BufRead, Write};
use std::cell::RefCell;
use std::rc::Rc;

struct Stream<R: BufRead> {
    reader: R,
    line: String,
    i: usize,
    line_num: usize,
    unread: Option<char>,
}

#[derive(Debug)]
pub enum End {
    Eof,
    Io(std::io::Error),
    Lisp(LispError),
}

impl fmt::Display for End {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use End::*;
        match self {
            Eof => write!(f, "eof"),
            Io(e) => write!(f, "{e}"),
            Lisp(e) => write!(f, "{e}"),
        }
    }
}

#[derive(Debug)]
pub enum LispError {
    Parse(String),
    Type(String),
    Env(String),
}

impl fmt::Display for LispError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use LispError::*;
        match self {
            Parse(e) => write!(f, "parse: {e}"),
            Type(e) => write!(f, "type: {e}"),
            Env(e) => write!(f, "env: {e}"),
        }
    }
}

fn symbol_start(c: char) -> bool {
    match c {
        '*'|'/'|'>'|'<'|'='|'?'|'!'|'-'|'+'|'A'..='Z'|'a'..='z' => true,
        _ => false,
    }
}

impl<R: BufRead> Stream<R> {
    fn new(reader: R) -> Self {
        Self { reader, line: String::new(), i: 0, line_num: 1, unread: None }
    }

    /// Ensure we have at least one more byte available in `self.line` at `self.i`.
    /// Returns false on EOF.
    fn fill_if_needed(&mut self) -> io::Result<bool> {
        if self.i < self.line.len() {
            return Ok(true);
        }

        self.line.clear();
        self.i = 0;

        let n = self.reader.read_line(&mut self.line)?;
        if n == 0 {
            return Ok(false); // EOF
        }

        Ok(true)
    }

    fn peek_char(&mut self) -> io::Result<Option<char>> {
        if let Some(c) = self.unread {
            return Ok(Some(c));
        }
        if !self.fill_if_needed()? {
            return Ok(None);
        }
        Ok(self.line[self.i..].chars().next())
    }

    fn read_char(&mut self) -> io::Result<Option<char>> {
        let c = if let Some(c) = self.unread.take() {
            c
        } else {
            let c = match self.peek_char()? {
                Some(c) => c,
                None => return Ok(None),
            };
            self.i += c.len_utf8();
            c
        };
        if c == '\n' {
            self.line_num += 1;
        }
        Ok(Some(c))
    }

    fn unread_char(&mut self, c: char) {
        assert!(self.unread.is_none(), "Stream only supports one unread char");
        if c == '\n' {
            self.line_num = self.line_num.saturating_sub(1);
        }
        self.unread = Some(c);
    }

    fn eat_whitespace(&mut self) -> io::Result<()> {
        while let Some(c) = self.peek_char()? {
            if c.is_whitespace() {
                _ = self.read_char()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn read_fixnum(&mut self, first: char) -> Result<LO, End> {
        assert!(first == '-' || first.is_digit(10));
        let is_negative = first == '-';

        let mut acc: i64 = if is_negative { 0 } else { (first as u8 - b'0') as i64 };
        while let Some(c) = self.peek_char().map_err(|e| End::Io(e))? {
            if let Some(d) = c.to_digit(10) {
                _ = self.read_char().map_err(|e| End::Io(e))?;
                acc = acc * 10 + d as i64;
            } else {
                break;
            }
        }

        if is_negative {
            acc *= -1;
        }

        Ok(LO::Fixnum(acc))
    }

    fn read_lo(&mut self) -> Result<LO, End> {
        self.eat_whitespace().map_err(|e| End::Io(e))?;

        let c = match self.read_char().map_err(|e| End::Io(e))? {
            Some(c) => c,
            None => return Err(End::Eof),
        };

        if c.is_ascii_digit() || c == '~' {
            return self.read_fixnum(if c == '~' { '-' } else { c });
        }

        if symbol_start(c) {
            let mut acc = c.to_string();
            loop {
                if let Some(nc) = self.read_char().map_err(|e| End::Io(e))? {
                    let is_delim = match nc {
                        '"'|'('|')'|'{'|'}'|';' => true,
                        nc => nc.is_whitespace(),
                    };
                    if is_delim {
                        self.unread_char(nc);
                        break;
                    } else {
                        acc.push(nc);
                        continue;
                    }
                }

                break;
            }
            return Ok(LO::Symbol(acc));
        }

        if c == '#' {
            match self.read_char().map_err(|e| End::Io(e))? {
                Some('t') => return Ok(LO::Bool(true)),
                Some('f') => return Ok(LO::Bool(false)),
                Some(_) | None => { },
            }
        }

        if c == '(' {
            let mut acc = LO::Nil;
            loop {
                self.eat_whitespace().map_err(|e| End::Io(e))?;
                let nc = self.read_char().map_err(|e| End::Io(e))?;
                if nc.is_none() {
                    return Err(End::Lisp(LispError::Parse("unexpected eof in list".to_string())));
                }

                let nc = nc.unwrap();

                if nc == ')' {
                    return reverse_list(acc).map_err(|e| End::Lisp(LispError::Parse(e)));
                }

                self.unread_char(nc);
                let car = self.read_lo()?;
                acc = LO::Pair(Box::new((car, acc)));
            }
        }

        if c == '\'' {
            return Ok(LO::Quote(Box::new(self.read_lo()?)));
        }

        let mut s = "unexpected char: ".to_string();
        s.push(c);
        Err(End::Lisp(LispError::Parse(s)))
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Env {
    items: Vec<(String, Rc<RefCell<Option<LO>>>)>,
}

impl Env {
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn bind(mut self, name: String, value: LO) -> Self {
        self.items.push((name, Rc::new(RefCell::new(Some(value)))));
        self
    }

    fn bind_rc(mut self, name: String, b: Rc<RefCell<Option<LO>>>) -> Self {
        self.items.push((name, b));
        self
    }

    fn make_loc() -> Rc<RefCell<Option<LO>>> {
        Rc::new(RefCell::new(None))
    }

    fn bind_list(mut self, names: Vec<String>, values: Vec<LO>) -> Self {
        for (n, v) in names.iter().zip(values.iter()) {
            self = self.bind(n.clone(), v.clone());
        }
        self
    }

    fn lookup(&self, name: &str) -> Result<LO, LispError> {
        if self.items.is_empty() {
            let s = "could not find '".to_string() + name + "' in env due to empty env";
            return Err(LispError::Env(s));
        }

        for (n, v) in self.items.iter().rev() {
            if name == n {
                match v.borrow().as_ref() {
                    Some(v_prime) => return Ok(v_prime.clone()),
                    None => {
                        let s = "'".to_string() + name
                            + "' evaluated to an unspecified value in env";
                        return Err(LispError::Env(s));
                    },
                }
            }
        }

        let s = "could not find '".to_string() + name + "' in env";
        return Err(LispError::Env(s));
    }

    fn basis() -> Env {
        use LO::Primitive;
        fn num_args(name: &str, n: usize, args: &[&LO]) -> Result<(), LispError> {
            if args.len() != n {
                let s = "'".to_string() + name + "' primitive takes " + &n.to_string()
                    + " arguments, found: " + &args.len().to_string();
                return Err(LispError::Type(s));
            }

            Ok(())
        }

        macro_rules! bin_fixnum_prim {
            ($name:literal, $ctor:path, $op:tt) => {
                Primitive($name.to_string(), |args| {
                    num_args($name, 2, args)?;
                    if let (LO::Fixnum(a), LO::Fixnum(b)) = (args[0], args[1]) {
                        Ok($ctor(a $op b))
                    } else {
                        let s = "'".to_string() + $name
                            + "' primitive takes integer arguments, found: '"
                            + &args[0].to_string() + "' and '" + &args[1].to_string() + "'";
                        Err(LispError::Type(s))
                    }
                })
            };
        }

        let prim_add = bin_fixnum_prim!("+", LO::Fixnum, +);
        let prim_sub = bin_fixnum_prim!("-", LO::Fixnum, -);
        let prim_mult = bin_fixnum_prim!("*", LO::Fixnum, *);

        let prim_eq = Primitive("eq".to_string(), |args| {
            num_args("eq", 2, args)?;
            Ok(LO::Bool(args[0] == args[1]))
        });

        let prim_pair = Primitive("pair".to_string(), |args| {
            num_args("pair", 2, args)?;
            Ok(LO::Pair(Box::new((args[0].clone(), args[1].clone()))))
        });


        let prim_list = Primitive("list".to_string(), |args| {
            fn prim_list(args: &[&LO]) -> LO {
                match args {
                    [] => LO::Nil,
                    [car, cdr @ ..] => LO::Pair(Box::new(((*car).clone(), prim_list(cdr)))),
                }
            }
            Ok(prim_list(args))
        });

        let prim_car = Primitive("car".to_string(), |args| {
            num_args("car", 1, args)?;
            if let LO::Pair(p) = args[0] {
                return Ok(p.0.clone());
            }

            let s = "'car' primitive expects a pair as argument, found: '".to_string()
                + &args[0].to_string() + "'";
            Err(LispError::Type(s))
        });

        let prim_cdr = Primitive("cdr".to_string(), |args| {
            num_args("car", 1, args)?;
            if let LO::Pair(p) = args[0] {
                return Ok(p.1.clone());
            }

            let s = "'cdr' primitive expects a pair as argument, found: '".to_string()
                + &args[0].to_string() + "'";
            Err(LispError::Type(s))
        });

        let prim_atomp = Primitive("atom?".to_string(), |args| {
            num_args("atom?", 1, args)?;
            if let LO::Pair(_) = args[0] {
                Ok(LO::Bool(false))
            } else {
                Ok(LO::Bool(true))
            }
        });

        let env = Env::new();
        let env = env.bind("+".to_string(), prim_add);
        let env = env.bind("eq".to_string(), prim_eq);
        let env = env.bind("-".to_string(), prim_sub);
        let env = env.bind("*".to_string(), prim_mult);
        let env = env.bind("pair".to_string(), prim_pair);
        let env = env.bind("list".to_string(), prim_list);
        let env = env.bind("car".to_string(), prim_car);
        let env = env.bind("cdr".to_string(), prim_cdr);
        let env = env.bind("atom?".to_string(), prim_atomp);
        env
    }

    fn to_lo(&self) -> LO {
        let los: Vec<LO> = self.items
            .iter()
            .map(|(n, v)|
                LO::Pair(Box::new((LO::Symbol(n.clone()), match v.borrow().as_ref() {
                    Some(lo) => lo.clone(),
                    None => LO::Symbol("#<unspecified value>".to_string()),
                })))
            ).collect();
        LO::list_to_pair(los)
    }
}

impl fmt::Display for Env {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let p = self.to_lo();
        write!(f, "{p}")
    }
}

/// Left-object
#[derive(Debug, PartialEq, Clone)]
enum LO {
    Fixnum(i64),
    Bool(bool),
    Symbol(String),
    Nil,
    Pair(Box<(LO, LO)>),
    Primitive(String, fn(&[&LO]) -> Result<LO, LispError>),
    Quote(Box<LO>),
    Closure(Vec<String>, Box<Expr>, Env),
}

fn reverse_list(mut xs: LO) -> Result<LO, String> {
    let mut out = LO::Nil;
    loop {
        match xs {
            LO::Nil => return Ok(out),
            LO::Pair(cell) => {
                let (car, cdr) = *cell;
                out = LO::Pair(Box::new((car, out)));
                xs = cdr;
            },
            _ => return Err("malformed list".to_string()),
        }
    }
}

impl LO {
    fn is_list(&self) -> bool {
        match self {
            LO::Nil => true,
            LO::Pair(cell) => cell.1.is_list(),
            _ => false,
        }
    }

    fn build_ast(&self) -> Result<Expr, LispError> {
        use LO::*;
        match self {
            Primitive(_, _) | Closure(_, _, _) => unreachable!(), // shouldn't happen at this stage.
            Symbol(s) => Ok(Expr::Var(s.clone())),
            Pair(_) if self.is_list() => {
                match self.pair_to_list().as_slice() {
                    [] => Err(LispError::Parse("poorly formed expression".to_string())),
                    [sym, cond, if_true, if_false] if matches!(sym, Symbol(s) if s == "if") =>
                        Ok(Expr::If(Box::new((cond.build_ast()?, if_true.build_ast()?, if_false.build_ast()?)))),
                    [sym, c1, c2] if matches!(sym, Symbol(s) if s == "and") =>
                        Ok(Expr::And(Box::new((c1.build_ast()?, c2.build_ast()?)))),
                    [sym, c1, c2] if matches!(sym, Symbol(s) if s == "or") =>
                        Ok(Expr::Or(Box::new((c1.build_ast()?, c2.build_ast()?)))),
                    [sym, func, args] if matches!(sym, Symbol(s) if s == "apply") =>
                        Ok(Expr::Apply(Box::new((func.build_ast()?, args.build_ast()?)))),
                    [sym, Symbol(n), e] if matches!(sym, Symbol(s) if s == "val") =>
                        Ok(Expr::Def(Box::new(Definition::Val(n.clone(), e.build_ast()?)))),
                    [sym, e] if matches!(sym, Symbol(s) if s == "quote") =>
                        Ok(Expr::Literal((*e).clone())),
                    [sym, ns, e] if ns.is_list() && matches!(sym, Symbol(s) if s == "lambda") => {
                        let formals: Result<Vec<String>, LispError> = ns
                            .pair_to_list()
                            .into_iter()
                            .map(|l| match l {
                                Symbol(s) => Ok(s.clone()),
                                _ => {
                                    let s = "arguments to lambda can only be symbols, found: '".to_string()
                                        + &l.to_string() + "'";
                                    Err(LispError::Type(s))
                                },
                            })
                            .collect();
                        let ast = e.build_ast()?;
                        Ok(Expr::Lambda(formals?, Box::new(ast)))
                    },
                    [sym, Symbol(n), ns, e] if matches!(sym, Symbol(s) if s == "define") => {
                        let formals: Result<Vec<String>, LispError> = ns
                            .pair_to_list()
                            .into_iter()
                            .map(|l| match l {
                                Symbol(s) => Ok(s.clone()),
                                _ => {
                                    let s = "arguments to lambda can only be symbols, found: '".to_string()
                                        + &l.to_string() + "'";
                                    Err(LispError::Type(s))
                                },
                            })
                            .collect();
                        let ast = e.build_ast()?;
                        Ok(Expr::Def(Box::new(Definition::Fun(n.clone(), formals?, ast))))
                    },
                    [func, args @ ..] => {
                        let mut values = Vec::with_capacity(args.len());
                        for arg in args {
                            values.push(arg.build_ast()?);
                        }
                        Ok(Expr::Call(Box::new((func.build_ast()?, values))))
                    },
                }
            },
            Fixnum(_) | Bool(_) | Nil | Pair(_) | Quote(_) => Ok(Expr::Literal(self.clone())),
        }
    }

    fn pair_to_list(&self) -> Vec<&LO> {
        let mut out = Vec::new();
        let mut p = self;
        loop {
            match p {
                LO::Pair(cell) => {
                    let (fst, snd) = cell.as_ref();
                    out.push(fst);
                    p = snd;
                },
                LO::Nil => break,
                _ => panic!("malformed list"),
            }
        }
        out
    }

    fn list_to_pair(xs: Vec<LO>) -> LO {
        let mut acc = LO::Nil;
        for lo in xs.into_iter().rev() {
            acc = LO::Pair(Box::new((lo, acc)));
        }
        acc
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Expr {
    Literal(LO),
    Var(String),
    If(Box<(Expr, Expr, Expr)>),
    And(Box<(Expr, Expr)>),
    Or(Box<(Expr, Expr)>),
    Apply(Box<(Expr, Expr)>),
    Call(Box<(Expr, Vec<Expr>)>),
    Lambda(Vec<String>, Box<Expr>),
    Def(Box<Definition>),
}

#[derive(Debug, PartialEq, Clone)]
enum Definition {
    Val(String, Expr),
    Fun(String, Vec<String>, Expr),
    Expr(Expr),
}

impl Expr {
    fn eval(&self, env: Env) -> Result<(LO, Env), LispError> {
        match self {
            Expr::Def(d) => Expr::eval_def(d, env),
            e => Ok((e.eval_expr(&env)?, env)),
        }
    }

    /// Returns the modified env.
    fn eval_def(def: &Definition, env: Env) -> Result<(LO, Env), LispError> {
        match def {
            Definition::Val(n, e) => {
                let v = e.eval_expr(&env)?;
                let env_prime = env.bind(n.clone(), v.clone());
                Ok((v, env_prime))
            },
            Definition::Fun(n, ns, e) => {
                let lambda = Expr::Lambda(ns.clone(), Box::new(e.clone()));
                let (formals, body, cl_env) = match lambda.eval_expr(&env)? {
                    LO::Closure(fs, body, env) => (fs, body, env),
                    lo => {
                        let s = "expected a closure to define a function, found: '".to_string()
                            + &lo.to_string() + "'";
                        return Err(LispError::Type(s));
                    }
                };
                let loc = Env::make_loc();
                let clo = LO::Closure(formals, body, cl_env.bind_rc(n.to_string(), loc.clone()));
                *loc.borrow_mut() = Some(clo.clone());
                Ok((clo, env.bind_rc(n.to_string(), loc)))
            },
            Definition::Expr(e) => Ok((e.eval_expr(&env)?, env)),
        }
    }

    /// Does not modify env, and returns the evaluated expression.
    fn eval_expr(&self, env: &Env) -> Result<LO, LispError> {
        use Expr::*;

        match self {
            Def(_) => unreachable!(),
            Literal(LO::Quote(b)) => Ok(*b.clone()),
            Literal(l) => Ok(l.clone()),
            Var(n) => env.lookup(&n),
            If(b) => match (*b).0.eval_expr(env)? {
                LO::Bool(true) => Ok((*b).1.eval_expr(env)?),
                LO::Bool(false) => Ok((*b).2.eval_expr(env)?),
                other => {
                    let s = "if statement condition did not resolve to a bool, found: '".to_string()
                        + &other.to_string() + "'";
                    Err(LispError::Type(s))
                },
            },
            And(b) => match ((*b).0.eval_expr(env)?, (*b).1.eval_expr(env)?) {
                | (LO::Bool(v1), LO::Bool(v2)) => Ok(LO::Bool(v1 && v2)),
                | (v1, v2) => {
                    let s = "and statement conditions did not resolve to bools, found: '".to_string()
                        + &v1.to_string() + "' and '" + &v2.to_string() + "'";
                    Err(LispError::Type(s))
                },
            },
            Or(b) => match ((*b).0.eval_expr(env)?, (*b).1.eval_expr(env)?) {
                | (LO::Bool(v1), LO::Bool(v2)) => Ok(LO::Bool(v1 || v2)),
                | (v1, v2) => {
                    let s = "or statement conditions did not resolve to bools, found: '".to_string()
                        + &v1.to_string() + "' and '" + &v2.to_string() + "'";
                    Err(LispError::Type(s))
                },
            },
            Apply(b) => {
                let f = (*b).0.eval_expr(env)?;
                let arg_list = (*b).1.eval_expr(env)?;
                if !arg_list.is_list() {
                    let s = "cannot apply a non-list '".to_string()
                        + &arg_list.to_string() + "' to a primitive";
                    return Err(LispError::Type(s));
                }

                let mut primed = Vec::new();
                for arg in arg_list.pair_to_list() {
                    let ast = arg.build_ast()?;
                    primed.push(ast.eval_expr(env)?);
                }
                let primed_ref: Vec<&LO> = primed.iter().collect();

                match f {
                    LO::Primitive(_, f) => f(primed_ref.as_slice()),
                    LO::Closure(ns, e, cl_env) => e.eval_expr(&cl_env.bind_list(ns, primed)),
                    _ => {
                        let s = "cannot apply to a non-primitive/closure: '".to_string()
                            + &f.to_string() + "'";
                        Err(LispError::Type(s))
                    },
                }
            },
            Call(b) => {
                if let (Expr::Var(name), true) = (&(*b).0, (*b).1.is_empty()) {
                    if name == "env" {
                        let los: Vec<LO> = env.items
                            .iter()
                            .map(|(n, v)|
                                LO::Pair(Box::new((LO::Symbol(n.clone()), match v.borrow().as_ref() {
                                    Some(lo) => lo.clone(),
                                    None => LO::Symbol("#<unspecified value>".to_string()),
                                })))
                            ).collect();
                        let env = LO::list_to_pair(los);
                        return Ok(env.clone());
                    }
                }

                let f = (*b).0.eval_expr(env)?;

                let args = &(*b).1;
                let mut primed = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    primed.push(arg.eval_expr(env)?);
                }
                let primed_ref: Vec<&LO> = primed.iter().collect();

                match f {
                    LO::Primitive(_, f) => f(primed_ref.as_slice()),
                    LO::Closure(ns, e, cl_env) => e.eval_expr(&cl_env.bind_list(ns, primed)),
                    _ => {
                        let s = "cannot call a non-primitive/closure: '".to_string()
                            + &f.to_string() + "'";
                        Err(LispError::Type(s))
                    },
                }
            },
            Lambda(ns, e) => Ok(LO::Closure(ns.clone(), e.clone(), env.clone())),
        }
    }
}

impl fmt::Display for LO {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use LO::*;

        match self {
            Fixnum(x) => write!(f, "{x}"),
            Bool(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            Symbol(s) => write!(f, "{s}"),
            Nil => write!(f, "nil"),
            Pair(b) => {
                write!(f, "(")?;
                if self.is_list() {
                    let mut p = b;
                    loop {
                        write!(f, "{}", p.0)?;
                        match &p.1 {
                            Pair(np) => {
                                p = &np;
                            },
                            Nil => break,
                            _ => panic!("malformed list"),
                        }
                        write!(f, " ")?;
                    }
                } else {
                    write!(f, "{} . {}", b.0, b.1)?;
                }
                write!(f, ")")
            },
            Primitive(name, _) => write!(f, "#<primitive:{name}>"),
            Quote(q) => write!(f, "'{}", *q),
            // TODO: clean this up
            Closure(_, _, _) => write!(f, "#<closure>"),
        }
    }
}

fn repl<R: BufRead>(stream: &mut Stream<R>, env: Env) -> io::Result<()> {
    use End::*;
    use LispError::*;

    let mut e = env;
    loop {
        print!("> ");
        _ = io::stdout().flush()?;

        let expr = match stream.read_lo() {
            Ok(lo) => lo,
            Err(Eof) => break,
            Err(Lisp(e)) => {
                println!("error: lisp: {e}");
                continue;
            }
            Err(Io(e)) => {
                println!("error: io: {e}");
                continue;
            },
        };

        let ast = match expr.build_ast() {
            Ok(a) => a,
            Err(l) => {
                println!("error: lisp: {l}");
                continue;
            },
        };

        let (result, env_prime) = match ast.eval(e.clone()) {
            Ok(x) => x,
            Err(l) => {
                println!("error: lisp: {l}");
                continue;
            },
        };
        e = env_prime;

        println!("{result}");
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let mut stream = Stream::new(io::stdin().lock());
    repl(&mut stream, Env::basis())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use io::Cursor;

    #[test]
    fn read_char() {
        let input = Cursor::new("hello  \n\t world");
        let mut s = Stream::new(input);

        assert_eq!(s.read_char().unwrap().unwrap(), 'h');
        assert_eq!(s.read_char().unwrap().unwrap(), 'e');
        assert_eq!(s.read_char().unwrap().unwrap(), 'l');
        assert_eq!(s.read_char().unwrap().unwrap(), 'l');
        assert_eq!(s.read_char().unwrap().unwrap(), 'o');
        assert_eq!(s.read_char().unwrap().unwrap(), ' ');
        assert_eq!(s.read_char().unwrap().unwrap(), ' ');

        assert_eq!(s.line_num, 1);
        assert_eq!(s.read_char().unwrap().unwrap(), '\n');
        assert_eq!(s.line_num, 2);

        assert_eq!(s.read_char().unwrap().unwrap(), '\t');
        assert_eq!(s.line_num, 2);

        assert_eq!(s.read_char().unwrap().unwrap(), ' ');
        assert_eq!(s.read_char().unwrap().unwrap(), 'w');
        assert_eq!(s.read_char().unwrap().unwrap(), 'o');
        assert_eq!(s.read_char().unwrap().unwrap(), 'r');
        assert_eq!(s.read_char().unwrap().unwrap(), 'l');
        assert_eq!(s.read_char().unwrap().unwrap(), 'd');
        assert_eq!(s.read_char().unwrap(), None); // eof
    }

    #[test]
    fn test_pair_to_list() {
        let input = Cursor::new("(1 2 3 4 5 350)");
        let mut s = Stream::new(input);

        let lo = s.read_lo().unwrap();
        let v = lo.pair_to_list();
        let exp: Vec<LO> = vec![1 as i64, 2, 3, 4, 5, 350].iter().map(|x| LO::Fixnum(*x)).collect();
        assert_eq!(v.len(), exp.len());

        for (i, val) in v.iter().enumerate() {
            assert_eq!(**val, exp[i]);
        }
    }

    #[test]
    fn unread_char() {
        let input = Cursor::new("ab\nc");
        let mut s = Stream::new(input);

        assert_eq!(s.read_char().unwrap().unwrap(), 'a');
        s.unread_char('a');
        assert_eq!(s.peek_char().unwrap().unwrap(), 'a');
        assert_eq!(s.read_char().unwrap().unwrap(), 'a');
        assert_eq!(s.read_char().unwrap().unwrap(), 'b');

        assert_eq!(s.line_num, 1);
        assert_eq!(s.read_char().unwrap().unwrap(), '\n');
        assert_eq!(s.line_num, 2);
        s.unread_char('\n');
        assert_eq!(s.line_num, 1);
        assert_eq!(s.peek_char().unwrap().unwrap(), '\n');
        assert_eq!(s.read_char().unwrap().unwrap(), '\n');
        assert_eq!(s.line_num, 2);

        assert_eq!(s.read_char().unwrap().unwrap(), 'c');
        assert_eq!(s.read_char().unwrap(), None);
    }

    #[test]
    fn parse_simples() {
        let input = Cursor::new("   12   \n15 340 #t #f ~90 hello_world bear-lisp");
        let mut s = Stream::new(input);

        assert_eq!(s.read_lo().unwrap(), LO::Fixnum(12));
        assert_eq!(s.line_num, 1);
        assert_eq!(s.read_lo().unwrap(), LO::Fixnum(15));
        assert_eq!(s.line_num, 2);
        assert_eq!(s.read_lo().unwrap(), LO::Fixnum(340));
        assert_eq!(s.read_lo().unwrap(), LO::Bool(true));
        assert_eq!(s.read_lo().unwrap(), LO::Bool(false));
        assert_eq!(s.read_lo().unwrap(), LO::Fixnum(-90));
        assert_eq!(s.read_lo().unwrap(), LO::Symbol("hello_world".to_string()));
        assert_eq!(s.read_lo().unwrap(), LO::Symbol("bear-lisp".to_string()));

        let input = Cursor::new("(1 2 hello world) (34 (35 some))");
        let mut s = Stream::new(input);
        assert_eq!(s.read_lo().unwrap().to_string(), "(1 2 hello world)");
        assert_eq!(s.read_lo().unwrap().to_string(), "(34 (35 some))");
    }

    #[test]
    fn eval() {
        let input = Cursor::new("(if #t (if #t 1 2) 3)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::new()).unwrap().0.to_string(), "1");

        let input = Cursor::new("(if #f (if #t 1 2) (if #t (list 34 35) 12))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "(34 35)");

        let env = Env::new();

        let input = Cursor::new("(val x #t)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        let (res, env) = ast.eval(env).unwrap();
        assert_eq!(res.to_string(), "#t");

        let input = Cursor::new("(val y (if x ~12 13))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        let (res, env) = ast.eval(env).unwrap();
        assert_eq!(res.to_string(), "-12");

        assert_eq!(env.lookup("x").unwrap(), LO::Bool(true));
        assert_eq!(env.lookup("y").unwrap(), LO::Fixnum(-12));
    }

    #[test]
    fn eval_basis() {
        let input = Cursor::new("(+ 12 13)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "25");

        let input = Cursor::new("(pair 12 13)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "(12 . 13)");

        let input = Cursor::new("(pair (pair 12 13) 14)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "((12 . 13) . 14)");

        let input = Cursor::new("(pair 12 (pair 13 14))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "(12 . (13 . 14))");

        let input = Cursor::new("(eq ((lambda (x) (+ x 1)) 10) 11)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "#t");

        let input = Cursor::new("(eq ((lambda (x) (+ x 1)) 10) 12)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();
        assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "#f");

        // NOTE: should stuff like this be allowed? currently the program experiences stack overflow
        // when trying to compare functions stored in an env.
        //
        // Also trying to implement alpha equivalence might be weird.
        // let input = Cursor::new("(eq (lambda (x) (x + 1)) (lambda (y) (y + 1)))");
        // let mut s = Stream::new(input);
        // let e = s.read_lo().unwrap();
        // let ast = e.build_ast().unwrap();
        // assert_eq!(ast.eval(Env::basis()).unwrap().0.to_string(), "#t");
    }

    #[test]
    fn eval_env_form() {
        let input = Cursor::new("(env)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let env = Env::basis();
        let (result, env_prime) = ast.eval(env.clone()).unwrap();
        assert_eq!(result.to_string(), env.to_string());
        assert_eq!(env_prime.to_string(), env.to_string());
    }

    #[test]
    fn eval_applications_and_quotes() {
        let input = Cursor::new("(apply + (list 13 14))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let env = Env::basis();
        let (result, _) = ast.eval(env).unwrap();
        assert_eq!(result, LO::Fixnum(27));

        let input = Cursor::new("(apply + '((if #t ~12 13) 14))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let env = Env::basis();
        let (result, _) = ast.eval(env).unwrap();
        assert_eq!(result, LO::Fixnum(2));
    }

    #[test]
    fn eval_lambda() {
        let env = Env::basis();

        let input = Cursor::new("(val add-one (lambda (x) (+ x 1)))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (_, env) = ast.eval(env).unwrap();

        let input = Cursor::new("(add-one 12)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (res, env) = ast.eval(env).unwrap();
        assert_eq!(res, LO::Fixnum(13));

        let input = Cursor::new("(val add-three (lambda (x) (add-one (add-one (add-one x)))))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (_, env) = ast.eval(env).unwrap();

        let input = Cursor::new("(add-three ~90)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (res, _) = ast.eval(env).unwrap();
        assert_eq!(res, LO::Fixnum(-87));
    }

    #[test]
    fn define_and_eval_function() {
        let env = Env::basis();

        let input = Cursor::new("(define f (x) (if (eq x 0) 1 (* x (f (+ x ~1)))))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (_, env) = ast.eval(env).unwrap();

        let input = Cursor::new("(f 4)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (res, env) = ast.eval(env).unwrap();
        assert_eq!(res, LO::Fixnum(24));

        let input = Cursor::new("(f 5)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (res, env) = ast.eval(env).unwrap();
        assert_eq!(res, LO::Fixnum(120));

        let input = Cursor::new("(f 6)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let ast = e.build_ast().unwrap();

        let (res, _) = ast.eval(env).unwrap();
        assert_eq!(res, LO::Fixnum(720));
    }
}
