# Natural Language to SQL (Python + NLP)

This mini project converts natural language requests into SQL queries.

## Features
- Built with Python
- Uses lightweight NLP techniques (regex tokenization + simple stemming)
- Supports:
  - table detection
  - column selection
  - aggregate functions (`count`, `sum`, `avg`, `min`, `max`)
  - basic `WHERE` conditions (`=`, `>`, `<`, `>=`, `<=`, `LIKE`)
  - `LIMIT` (`top`, `first`, `limit`)

## Project structure
- `/home/runner/work/my-portfolio/my-portfolio/nl_to_sql/main.py` — converter CLI
- `/home/runner/work/my-portfolio/my-portfolio/nl_to_sql/schema.json` — sample schema
- `/home/runner/work/my-portfolio/my-portfolio/nl_to_sql/requirements.txt` — dependencies

## Setup
```bash
cd /home/runner/work/my-portfolio/my-portfolio/nl_to_sql
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py "show name and email from users where age greater than 18"
```

Example output:
```sql
SELECT name, email FROM users WHERE age > 18;
```

More examples:
```bash
python main.py "how many users are there"
python main.py "show average amount from orders"
python main.py "list all products where category contains electronics top 5"
```
