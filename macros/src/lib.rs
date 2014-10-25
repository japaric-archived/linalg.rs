#![deny(warnings)]
#![feature(if_let, macro_rules, plugin_registrar, slicing_syntax)]

extern crate rustc;
extern crate syntax;

use rustc::plugin::registry::Registry;
use std::ptr;
use syntax::ast::{ExprBox, Inherited, LitInt, Plus, TTTok, TokenTree, TyVec, UnsuffixedIntLit};
use syntax::codemap::Span;
use syntax::ext::base::{DummyResult, ExtCtxt, MacExpr, MacResult, NormalTT};
use syntax::ext::build::AstBuilder;
use syntax::parse::token::{mod, COMMA, SEMI};

#[macro_export]
/// Creates an owned matrix from its arguments
///
/// - Each argument is an element of the matrix
/// - Semicolons `;` are used to separate rows
/// - Trailing semicolons are allowed
/// - Commas `,` are used to separate columns
/// - Trailing commas are *not* allowed
/// - If the matrix contains a single row, the macro will return a `Row`
/// - If the matrix contains a single column, the macro will return a `Col`
/// - Otherwise, the macro will return `Mat`
///
/// # Expansion
///
/// There are three possible expansions:
///
/// - Row vector
///
/// ``` ignore
/// mat![0, 1, 2]
/// ```
///
/// expands into:
///
/// ``` ignore
/// Row::new(vec![0, 1, 2])
/// ```
///
/// - Column vector
///
/// ``` ignore
/// // NB Newlines are *not* required
/// mat![
///     0;
///     1;
///     2;  // This semicolon can be removed if desired
/// ]
/// ```
///
/// expands into:
///
/// ``` ignore
/// Col::new(vec![0, 1, 2])
/// ```
///
/// - Matrix
///
/// ``` ignore
/// mat![
///     0, 1, 2;
///     3, 4, 5;
/// ]
/// ```
///
/// expands into:
///
/// ``` ignore
/// Mat::new(vec![0, 3, 1, 4, 3, 5], 2)
/// ```
///
/// Note that the order of the arguments have changed because matrices are stored in column-major
/// order
macro_rules! mat {
    ($($($elem:expr),+);+;) => { /* syntax extension */ }
}

#[plugin_registrar]
#[doc(hidden)]
pub fn plugin_registrar(r: &mut Registry) {
    r.register_syntax_extension(token::intern("mat"), NormalTT(box expand_mat, None));
}

fn expand_mat<'cx>(
    cx: &'cx mut ExtCtxt,
    sp: Span,
    tts: &[TokenTree],
) -> Box<MacResult + 'cx> {
    fn at_semicolons(tt: &TokenTree) -> bool {
        if let TTTok(_, SEMI) = *tt {
            true
        } else {
            false
        }
    }

    fn at_commas(tt: &TokenTree) -> bool {
        if let TTTok(_, COMMA) = *tt {
            true
        } else {
            false
        }
    }

    let rows = {
        let mut rows = tts.split(at_semicolons).collect::<Vec<_>>();
        // look for trailing semicolon
        if rows.last().map(|tts| tts.is_empty()) == Some(true) {
            rows.pop();
        }
        rows
    };

    let nrows = rows.len();

    let matrix = {
        let mut matrix = Vec::with_capacity(nrows);
        for (r, row) in rows.into_iter().enumerate() {
            let elems = row.split(at_commas).collect::<Vec<_>>();
            // look for trailing commas
            if elems.last().map(|tts| tts.is_empty()) == Some(true) {
                let err_msg = format!("row {}: trailing comma not allowed", r);
                cx.span_err(sp, err_msg[]);
                return DummyResult::expr(sp);
            }
            matrix.push(elems);
        }
        matrix
    };

    let ncols = {
        let mut cols_per_row = matrix.iter().map(|row| row.len());

        let ncols = match cols_per_row.next() {
            Some(ncols) => ncols,
            None => {
                cx.span_err(sp, "Empty matrix");
                return DummyResult::expr(sp);
            },
        };

        for (i, cols_per_row) in cols_per_row.enumerate() {
            if cols_per_row != ncols {
                let err_msg = format!(
                    "row {}: expected {} columns, but found {}",
                    i + 1,
                    ncols,
                    cols_per_row);
                cx.span_err(sp, err_msg[]);
                return DummyResult::expr(sp);
            }
        }

        ncols
    };

    for (r, row) in matrix.iter().enumerate() {
        for (c, elem) in row.iter().enumerate() {
            if elem.is_empty() {
                cx.span_err(sp, format!("no element found at ({}, {})", r, c)[]);
                return DummyResult::expr(sp);
            }
        }
    }

    let nelems = nrows * ncols;
    let mut elems = Vec::with_capacity(nelems);
    unsafe { elems.set_len(nelems) }

    for (r, row) in matrix.into_iter().enumerate() {
        for (c, elem) in row.into_iter().enumerate() {
            let dst = elems.get_mut(c * nrows + r);

            unsafe { ptr::write(dst, cx.new_parser_from_tts(elem).parse_expr()) }
        }
    }

    // XXX Ugh, how do I call the existing `vec!` macro here? I'm reinventing the wheel :-(
    let vec = {
        let uses = vec![{
            let segments = vec![
                token::str_to_ident("std"),
                token::str_to_ident("slice"),
                token::str_to_ident("BoxedSlice"),
            ];

            cx.view_use_simple(sp, Inherited, cx.path_global(sp, segments))
        }, {
            let segments = vec![
                token::str_to_ident("std"),
                token::str_to_ident("boxed"),
                token::str_to_ident("HEAP"),
            ];

            cx.view_use_simple(sp, Inherited, cx.path_global(sp, segments))
        }];

        let stmts = vec![{
            let ident = token::str_to_ident("xs");
            let expr = {
                let heap = cx.expr_ident(sp, token::str_to_ident("HEAP"));
                let array = cx.expr_vec(sp, elems);

                cx.expr(sp, ExprBox(heap, array))
            };
            let ty = {
                let path = {
                    let segments = vec![
                        token::str_to_ident("std"),
                        token::str_to_ident("boxed"),
                        token::str_to_ident("Box"),
                    ];
                    let ty = cx.ty(sp, TyVec(cx.ty_infer(sp)));

                    cx.path_all(sp, true, segments, vec![], vec![ty])
                };
                cx.ty_path(path, None)
            };

            cx.stmt_let_typed(sp, false, ident, ty, expr)
        }];

        let expr = Some({
            let receiver = cx.expr_ident(sp, token::str_to_ident("xs"));
            let method = token::str_to_ident("into_vec");
            let args = vec![];

            cx.expr_method_call(sp, receiver, method, args)
        });

        cx.expr_block(cx.block_all(sp, uses, stmts, expr))
    };

    if nrows == 1 {
        let fn_name = {
            let segments = vec![
                token::str_to_ident("linalg"),
                token::str_to_ident("Row"),
                token::str_to_ident("new"),
            ];

            cx.expr_path(cx.path_global(sp, segments))
        };
        let args = vec![vec];

        MacExpr::new(cx.expr_call(sp, fn_name, args))
    } else if ncols == 1 {
        let fn_name = {
            let segments = vec![
                token::str_to_ident("linalg"),
                token::str_to_ident("Col"),
                token::str_to_ident("new"),
            ];

            cx.expr_path(cx.path_global(sp, segments))
        };
        let args = vec![vec];

        MacExpr::new(cx.expr_call(sp, fn_name, args))
    } else {
        let fn_name = {
            let segments = vec![
                token::str_to_ident("linalg"),
                token::str_to_ident("Mat"),
                token::str_to_ident("new"),
            ];

            cx.expr_path(cx.path_global(sp, segments))
        };
        let args = vec![vec, cx.expr_lit(sp, LitInt(nrows as u64, UnsuffixedIntLit(Plus)))];

        MacExpr::new(cx.expr_call(sp, fn_name, args))
    }
}
