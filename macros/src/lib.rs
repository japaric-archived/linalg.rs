#![deny(warnings)]
#![feature(macro_rules, plugin_registrar, slicing_syntax)]

extern crate rustc;
extern crate syntax;

use rustc::plugin::registry::Registry;
use std::ptr;
use syntax::ast::{
    DUMMY_NODE_ID, Block, CompilerGenerated, ExprBox, Inherited, LitInt, Plus, TtToken, TokenTree,
    UnsafeBlock, UnsuffixedIntLit,
};
use syntax::codemap::Span;
use syntax::ext::base::{DummyResult, ExtCtxt, MacExpr, MacResult, NormalTT};
use syntax::ext::build::AstBuilder;
use syntax::parse::token::{self, Comma, Semi};
use syntax::ptr::P;

#[macro_export]
/// Creates an owned matrix from its arguments
///
/// - Each argument is an element of the matrix
/// - Semicolons `;` are used to separate rows
/// - Trailing semicolons are allowed
/// - Commas `,` are used to separate columns
/// - Trailing commas are *not* allowed
/// - If the matrix contains a single row, the macro will return `RowVec`
/// - If the matrix contains a single column, the macro will return `ColVec`
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
/// RowVec::new(box [0, 1, 2])
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
/// ColVec::new(box [0, 1, 2])
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
/// unsafe { Mat::from_parts(box [0, 3, 1, 4, 3, 5], (2, 3)) }
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
        if let TtToken(_, Semi) = *tt {
            true
        } else {
            false
        }
    }

    fn at_commas(tt: &TokenTree) -> bool {
        if let TtToken(_, Comma) = *tt {
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
            let dst = &mut elems[c * nrows + r];

            unsafe { ptr::write(dst, cx.new_parser_from_tts(elem).parse_expr()) }
        }
    }

    let vec = {
        let uses = vec![{
            let segments = vec![
                cx.ident_of("std"),
                cx.ident_of("boxed"),
                cx.ident_of("HEAP"),
            ];

            cx.view_use_simple(sp, Inherited, cx.path_global(sp, segments))
        }];

        let stmts = vec![];

        let expr = {
            let heap = cx.expr_ident(sp, cx.ident_of("HEAP"));
            let array = cx.expr_vec(sp, elems);

            Some(cx.expr(sp, ExprBox(Some(heap), array)))
        };

        cx.expr_block(cx.block_all(sp, uses, stmts, expr))
    };

    if nrows == 1 {
        let fn_name = {
            let segments = vec![
                cx.ident_of("linalg"),
                cx.ident_of("RowVec"),
                cx.ident_of("new"),
            ];

            cx.expr_path(cx.path_global(sp, segments))
        };
        let args = vec![vec];

        MacExpr::new(cx.expr_call(sp, fn_name, args))
    } else if ncols == 1 {
        let fn_name = {
            let segments = vec![
                cx.ident_of("linalg"),
                cx.ident_of("ColVec"),
                cx.ident_of("new"),
            ];

            cx.expr_path(cx.path_global(sp, segments))
        };
        let args = vec![vec];

        MacExpr::new(cx.expr_call(sp, fn_name, args))
    } else {
        let fn_name = {
            let segments = vec![
                cx.ident_of("linalg"),
                cx.ident_of("Mat"),
                cx.ident_of("from_parts"),
            ];

            cx.expr_path(cx.path_global(sp, segments))
        };
        let size = cx.expr_tuple(sp, vec![
            cx.expr_lit(sp, LitInt(nrows as u64, UnsuffixedIntLit(Plus))),
            cx.expr_lit(sp, LitInt(ncols as u64, UnsuffixedIntLit(Plus))),
        ]);
        let args = vec![vec, size];

        let unsafe_block = cx.expr_block(P(Block {
            view_items: vec![],
            stmts: vec![],
            expr: Some(cx.expr_call(sp, fn_name, args)),
            id: DUMMY_NODE_ID,
            rules: UnsafeBlock(CompilerGenerated),
            span: sp,
        }));

        MacExpr::new(unsafe_block)
    }
}
