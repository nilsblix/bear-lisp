use std::fmt;
use std::io::{self, BufRead, Write};

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
            Parse(e) => write!(f, "{e}"),
            Type(e) => write!(f, "{e}"),
            Env(e) => write!(f, "{e}"),
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

        let mut s = "unexpected char: ".to_string();
        s.push(c);
        Err(End::Lisp(LispError::Parse(s)))
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

fn is_list(xs: &LO) -> bool {
    match xs {
        LO::Nil => true,
        LO::Pair(cell) => is_list(&cell.1),
        _ => false,
    }
}

fn pair_to_list(xs: &LO) -> Vec<&LO> {
    let mut out = Vec::new();
    let mut p = xs;
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

/// env is a list (i.e a pair of pair of etc... with nil at the end), therefore bind creates a new
/// end with puts the (name, value) pair at the front of the env.
///
/// Ex:
/// ```
/// bind("hello", 12, (("world" . 13) ("var" 90))) -> (("hello" . 12) ("world" . 13) ("var" 90))
/// ```
fn bind(name: String, value: LO, env: LO) -> LO {
    use LO::*;
    let binding = Pair(Box::new((Symbol(name), value)));
    Pair(Box::new((binding, env)))
}

fn lookup<'a>(name: &str, env: &'a LO) -> Option<&'a LO> {
    use LO::*;

    let Pair(env_cell) = env else { return None; };
    let (binding, rest) = env_cell.as_ref();

    if let Pair(binding_cell) = binding {
        let (key, value) = binding_cell.as_ref();
        return match key {
            Symbol(k) if k == name => Some(value),
            _ => lookup(name, rest),
        }
    }

    if let Primitive(n, _) = binding {
        if n == name {
            return Some(binding);
        } else {
            return lookup(name, rest);
        }
    }

    None
}

impl LO {
    fn cons_from_vec(v: Vec<LO>) -> LO {
        let mut acc = LO::Nil;
        for lo in v.into_iter().rev() {
            acc = LO::Pair(Box::new((lo, acc)));
        }
        acc
    }

    fn eval_args(args: &[&LO], env: LO) -> Result<(Vec<LO>, LO), LispError> {
        let mut values = Vec::with_capacity(args.len());
        let mut env_cur = env;
        for arg in args {
            let (val, env_next) = arg.eval(env_cur)?;
            values.push(val);
            env_cur = env_next;
        }
        Ok((values, env_cur))
    }

    /// Returns the result and a modified env, in case the evaluation has side-effects.
    fn eval(&self, env: LO) -> Result<(LO, LO), LispError> {
        use LO::*;

        match self {
            Symbol(name) => {
                let value = match lookup(name, &env) {
                    Some(x) => x,
                    None => {
                        let mut s = "did not find '".to_string();
                        s += &name.to_string();
                        s += "' in env";
                        return Err(LispError::Env(s));
                    },
                };

                Ok((value.clone(), env))
            }
            Pair(_) if is_list(self) => {
                match pair_to_list(self).as_slice() {
                    [] => Ok((Nil, env)),
                    [sym, cond, if_true, if_false] if matches!(sym, Symbol(s) if s == "if") => {
                        let (cond_val, _) = cond.eval(env.clone())?;
                        match cond_val {
                            Bool(true) => if_true.eval(env),
                            Bool(false) => if_false.eval(env),
                            _ => {
                                let mut s = "expected bool in if condition, found: ".to_string();
                                s += &cond_val.to_string();
                                Err(LispError::Type(s))
                            },
                        }
                    },
                    [sym, name, val] if matches!(sym, Symbol(s) if s == "val") => {
                        let (val_prime, env) = val.eval(env)?;
                        let env_prime = bind(name.to_string(), val_prime.clone(), env);
                        Ok((val_prime, env_prime))
                    },

                    [sym] if matches!(sym, Symbol(s) if s == "env") => Ok((env.clone(), env)),
                    [lhs, args @ ..] => {
                        let (func, env) = lhs.eval(env)?;
                        match func {
                            Primitive(_, f) => {
                                let (evaluated, env) = LO::eval_args(args, env)?;
                                let arg_refs: Vec<&LO> = evaluated.iter().collect();
                                Ok((f(&arg_refs)?, env))
                            },
                            _ => Ok((self.clone(), env)),
                        }
                    },
                }
            },
            Fixnum(_) | Bool(_) | Nil | Primitive(_, _) | Pair(_) => Ok((self.clone(), env)),
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
                if is_list(self) {
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
        }
    }
}

fn basis() -> LO {
    use LO::Primitive;
    fn num_args(name: &str, n: usize, args: &[&LO]) -> Result<(), LispError> {
        if args.len() != n {
            let s = name.to_string() + " takes " + &n.to_string()
                + " arguments, found: " + &args.len().to_string();
            return Err(LispError::Type(s));
        }

        Ok(())
    }

    let plus = Primitive("+".to_string(), |args| {
        num_args("+", 2, args)?;

        if let (LO::Fixnum(a), LO::Fixnum(b)) = (args[0], args[1]) {
            Ok(LO::Fixnum(a + b))
        } else {
            let s = "'+' takes integer arguments, found: '".to_string()
                + &args[0].to_string() + "' and '" + &args[1].to_string();
            Err(LispError::Type(s))
        }
    });

    let pair = Primitive("pair".to_string(), |args| {
        num_args("pair", 2, args)?;
        Ok(LO::Pair(Box::new((args[0].clone(), args[1].clone()))))
    });

    LO::cons_from_vec(vec![pair, plus])
}

fn repl<R: BufRead>(stream: &mut Stream<R>, env: LO) -> io::Result<()> {
    let mut e = env;
    loop {
        print!("> ");
        _ = io::stdout().flush()?;

        let expr = match stream.read_lo() {
            Ok(lo) => lo,
            Err(End::Eof) => break,
            Err(End::Lisp(e)) => {
                println!("error: lisp: {e}");
                continue;
            }
            Err(End::Io(e)) => {
                println!("error: io: {e}");
                continue;
            },
        };

        let (result, env_prime) = match expr.eval(e.clone()) {
            Ok(x) => x,
            Err(e) => {
                println!("error: eval: {e}");
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
    repl(&mut stream, basis())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_char() {
        use io::Cursor;
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
        use io::Cursor;
        let input = Cursor::new("(1 2 3 4 5 350)");
        let mut s = Stream::new(input);

        let lo = s.read_lo().unwrap();
        let v = pair_to_list(&lo);
        let exp: Vec<LO> = vec![1 as i64, 2, 3, 4, 5, 350].iter().map(|x| LO::Fixnum(*x)).collect();
        assert_eq!(v.len(), exp.len());

        for (i, val) in v.iter().enumerate() {
            assert_eq!(**val, exp[i]);
        }
    }

    #[test]
    fn unread_char() {
        use io::Cursor;
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
        use io::Cursor;
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
    fn test_cons_from_vec() {
        let a = LO::Fixnum(45);
        let b = LO::Bool(false);
        let c = LO::Symbol("hello-world".to_string());
        assert_eq!(LO::cons_from_vec(vec![a, b, c]).to_string(), "(45 #f hello-world)")
    }

    #[test]
    fn eval() {
        use io::Cursor;

        let input = Cursor::new("(if #t (if #t 1 2) 3)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        assert_eq!(e.eval(LO::Nil).unwrap().0.to_string(), "1");

        let input = Cursor::new("(if #f (if #t 1 2) (if #t (34 35) 12))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        assert_eq!(e.eval(LO::Nil).unwrap().0.to_string(), "(34 35)");

        let env = LO::Nil;

        let input = Cursor::new("(val x #t)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let (res, env) = e.eval(env).unwrap();
        assert_eq!(res.to_string(), "#t");

        let input = Cursor::new("(val y (if x ~12 13))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        let (res, env) = e.eval(env).unwrap();
        assert_eq!(res.to_string(), "-12");

        assert_eq!(lookup("x", &env), Some(&LO::Bool(true)));
        assert_eq!(lookup("y", &env), Some(&LO::Fixnum(-12)));
    }

    #[test]
    fn eval_basis() {
        use io::Cursor;

        let input = Cursor::new("(+ 12 13)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        assert_eq!(e.eval(basis()).unwrap().0.to_string(), "25");

        let input = Cursor::new("(pair 12 13)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        assert_eq!(e.eval(basis()).unwrap().0.to_string(), "(12 . 13)");

        let input = Cursor::new("(pair (pair 12 13) 14)");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        assert_eq!(e.eval(basis()).unwrap().0.to_string(), "((12 . 13) . 14)");

        let input = Cursor::new("(pair 12 (pair 13 14))");
        let mut s = Stream::new(input);
        let e = s.read_lo().unwrap();
        assert_eq!(e.eval(basis()).unwrap().0.to_string(), "(12 . (13 . 14))");
    }
}
