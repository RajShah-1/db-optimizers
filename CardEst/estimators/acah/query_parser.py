# query_parser.py
"""Hacky regex-based SQL parser for ACAH prototype."""
import re
from collections import defaultdict

# --- Parsing functions --- 
# These are kept functionally the same as V3 placeholder version 
# with minor robustness tweaks. A proper AST parser is recommended.

def extract_tables(query):
    """Extracts table names and aliases using regex (basic)."""
    alias_map = {}
    # Look for FROM/JOIN clauses before WHERE
    processed_query = query.lower().split(' where ')[0] 
    
    # Find standard 'table [as] alias' patterns
    matches = re.findall(r'(?:from|join)\s+([a-z0-9_."]+)(?:\s+(?:as\s+)?([a-z0-9_."]+))?', processed_query)
    
    # Handle comma-separated tables after FROM more carefully
    from_clause_match = re.search(r'from\s+(.*?)(?:join|where|group by|order by|limit|;|$)', processed_query, re.DOTALL)
    if from_clause_match:
        from_tables_part = from_clause_match.group(1).strip()
        # Split by comma, respecting potential spaces around commas
        parts = [p.strip() for p in re.split(r'\s*,\s*', from_tables_part)]
        for part in parts:
            sub_parts = part.split()
            if len(sub_parts) > 0:
                 table_name = sub_parts[0].strip('"') # Remove potential quotes
                 # Alias is the last part, unless it's a keyword
                 alias = sub_parts[-1].strip('"') if len(sub_parts) > 1 and sub_parts[-1].lower() not in ['on','using','inner','outer','left','right','full'] else table_name
                 # Add if not already mapped
                 if alias not in alias_map: alias_map[alias] = table_name

    # Add tables found by the main regex (can override comma-sep if alias is reused)
    for match in matches:
        table_name, alias = match[0].strip('"'), match[1]
        if not alias: alias = table_name
        else: alias = alias.strip('"')
        alias_map[alias] = table_name # Overwrite allows JOIN syntax to dominate

    # Basic fallback for single table 'SELECT ... FROM table'
    if not alias_map: 
        select_from_match = re.search(r'select.*?from\s+([a-z0-9_."]+)', query, re.IGNORECASE)
        if select_from_match:
             table_name = select_from_match.group(1).strip('"')
             alias_map[table_name] = table_name
             
    return alias_map

def extract_joins(query, alias_map):
    """Extracts 'alias.col = alias.col' join conditions using regex."""
    join_conditions = []
    # Regex finds alias.col = alias.col allowing quotes
    join_matches = re.findall(r'([a-zA-Z0-9_."]+)\.([a-zA-Z0-9_."]+)\s*=\s*([a-zA-Z0-9_."]+)\.([a-zA-Z0-9_."]+)', query)
    for alias1, col1, alias2, col2 in join_matches:
        a1, c1 = alias1.strip('"'), col1.strip('"')
        a2, c2 = alias2.strip('"'), col2.strip('"')
        table1 = alias_map.get(a1)
        table2 = alias_map.get(a2)
        if table1 and table2: join_conditions.append((table1, c1, table2, c2))
    return join_conditions

def extract_predicates(query, alias_map, column_types_ref):
    """Extracts 'alias.col OP value' or 'alias.col IS [NOT] NULL' predicates using regex."""
    predicates = []
    # Combined regex for comparisons and IS NULL/NOT NULL
    predicate_matches = re.findall(
        # Group 1: Alias, Group 2: Column
        r'([a-zA-Z0-9_."]+)\.([a-zA-Z0-9_."]+)\s*' 
        # Group 3,4: Ops like =, >, LIKE etc. and the value OR Group 5: IS NULL/IS NOT NULL
        r'(?:([<>=!]+|LIKE)\s*(\?|\%?\'.*?\'%?|[0-9.-]+)|(IS(?:\s+NOT)?\s+NULL))',
        query, re.IGNORECASE
    )
    for match in predicate_matches:
         alias, column, operator, value_str, is_null_op = match
         a, c = alias.strip('"'), column.strip('"')
         table = alias_map.get(a)
         if not table or c not in column_types_ref.get(table, {}): continue # Skip if table/col unknown

         if is_null_op: # Handle IS NULL / IS NOT NULL
             operator = is_null_op.upper().replace(" ", "")
             value = None
         elif operator: # Handle comparison operators
             operator = operator.upper().replace(" ", "")
             value = value_str.strip()
             is_string_literal = value.startswith("'") and value.endswith("'")
             if is_string_literal: value = value[1:-1] # Remove quotes
             elif value == '?': value = 0 # Placeholder default
             else:
                 try: value = float(value) # Assume numeric otherwise
                 except ValueError: continue # Skip if value is not quoted string or number
         else: continue # Skip if parsing failed

         predicates.append((table, c, operator, value))
    return predicates

def parse_query(query, column_types_ref):
    """Top-level parsing function using regex helpers."""
    query = query.replace('\n', ' ').replace('\t', ' ') # Basic normalization
    alias_map = extract_tables(query)
    if not alias_map:
        print(f"Warning: Could not parse tables from query: {query[:100]}...")
        return {}, [], [] 
        
    join_conditions = extract_joins(query, alias_map)
    predicates = extract_predicates(query, alias_map, column_types_ref)
    return alias_map, join_conditions, predicates
