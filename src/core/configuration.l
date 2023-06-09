/*
    Copyright (C) 2012  EPFL (Ecole Polytechnique Fédérale de Lausanne)
    Copyright (C) 2012  Nicolas Bourdaud <nicolas.bourdaud@gmail.com>
    Laboratory CNBI (Chair in Non-Invasive Brain-Machine Interface)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
%option yylineno reentrant never-interactive batch bison-bridge
%option noyywrap noyyalloc noyyrealloc noyyfree nounput noinput
%option prefix="cf"
%option header-file="configuration.lex.h"
%option outfile="configuration.lex.c"
%{
#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <string.h>
#include "confparser.h"
#include "configuration.tab.h"
%}

%x SQUOTE DQUOTE

%%
%{
	char str_buf[TOKEN_MAXLEN+1];
	int slen = 0;
	struct cfdata* pp = yyextra;
%}

 /* keywords */
"mapping"		{ return DEFMAP; }
"endmapping"		{ return ENDMAP; }

 /* operators */
"="|"|"			{ return yytext[0]; }

"\n"|"\r\n"|";"		{ return EOL; }
#[^\n]*		/* skip comment */;
[ \t]+          /* ignore whitespace */;

[a-zA-Z0-9_.:+-/]*	{
				yylval->str = cfd_push_string(pp, yytext);
				return (yylval->str) ? WORD : -1;
			}

 /* Single and double quote string literal */
'			{ BEGIN(SQUOTE); slen = 0; }
\"			{ BEGIN(DQUOTE); slen = 0; }
<SQUOTE,DQUOTE>\n	return -1;

<SQUOTE>[^'\n]+		append_string(str_buf, &slen, yytext);
<DQUOTE>[^\\\n\"]+	append_string(str_buf, &slen, yytext);
<DQUOTE>{
	"\\n"		append_char(str_buf, &slen, '\n');
	"\\\""		append_char(str_buf, &slen, '\"');
	"\\\\"		append_char(str_buf, &slen, '\\');
}

<DQUOTE>\"		{
				BEGIN(INITIAL);
				str_buf[slen] = '\0';
				yylval->str = cfd_push_string(pp, str_buf);
				return (yylval->str) ? WORD : -1;
			}
<SQUOTE>'		{
				BEGIN(INITIAL);
				str_buf[slen] = '\0';
				yylval->str = cfd_push_string(pp, str_buf);
				return (yylval->str) ? WORD : -1;
			}

%%


/**********************************************************
 * reimplement these functions to switch off the warnings *
 **********************************************************/
LOCAL_FN
void *cfalloc (yy_size_t  size , yyscan_t yyscanner)
{
	(void)yyscanner;
	return malloc(size);
}


LOCAL_FN
void *cfrealloc  (void * ptr, yy_size_t  size , yyscan_t yyscanner)
{
	(void)yyscanner;
	return realloc(ptr, size);
}


LOCAL_FN
void cffree (void * ptr , yyscan_t yyscanner)
{
	(void)yyscanner;
	free(ptr);
}

