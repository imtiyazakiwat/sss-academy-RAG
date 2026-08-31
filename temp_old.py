"""
Extracts and structures all Q&A pairs from SSS Academy Notes (pdf_extracted.txt) into qa_data.json.
Preserves original wording, grammar, examples, and tone exactly.
"""

import json
import re

def create_qa_dataset():
    qa_list = []

    def add_qa(question, answer, topic):
        qa_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "topic": topic.strip()
        })

    # Page 1: Self Introduction & Roles
    add_qa(
        "Tell me about yourself / Self Introduction",
        """Hi,
As you know my name is ABCD I am having 4.2 years of experience in IT industry, in that 4years as ETL tester and remaining as manual tester.
Recently I have worked in HCL Technologies, Bengaluru and worked on Menards Retail Domain project, the client name is Menard data warehouse from United States.
In that project,
• I have used Oracle as Database,
• SQL as to write queries and to validate data,
• Informatica as ETL Tool,
• HP ALM as Project Management Tool (Bug Tracking Tool). (Application Lifecycle Management)
• TOAD is used as data accessing.
• I am very good to write SQL queries.
• I am a Quick learner, Team Player,
• Willing to learn new technologies.

And this is about my-self introduction.""",
        "Self Introduction"
    )

    add_qa(
        "Why should we hire you? / Why are you a good fit for this ETL Tester role?",
        """Why you should hire me:

• I have 4.2 years of solid hands-on experience in ETL and Data Warehouse testing, specifically on enterprise retail DWH projects.
• I have strong SQL skills to write complex queries for source-to-target mapping, MINUS validations, duplicate checks, and SCD Type 2 history testing.
• In my previous project at HCL, I worked daily with Informatica PowerCenter, Oracle DB, TOAD, and Unix shell commands.
• I have complete end-to-end testing lifecycle knowledge—from reviewing STM documents, writing test cases in HP ALM, raising defects to developers, and sending Daily Status Reports (DSR).
• I am a proactive team player and a fast learner who can hit the ground running with minimal hand-holding.""",
        "HR & General Interview"
    )

    add_qa(
        "What are your roles and responsibilities in your project as an ETL Software Testing Engineer?",
        """The Role in my last Project as Software Testing Engineer.
And responsibilities are:
• First I go through the technical documents such as source target mapping (STM).
• Then I have understood the client requirement and come up with high level scenario.
• Then I have developed the test cases and those test cases are undergone pear review.
• Once the test cases approved by Team Lead, we have uploaded in HP ALM.
• Once build deployed in testing environment, we start executing test cases.
• If any deviation between actual and expected result then I have raise the bug report to particular developer and I have follow-up until to it has resolved.
• I also assist team lead to produce test plans.
• Even I have part of all calls like scrum call, client call, Standup call, weekly call and monthly call.
• At end of the day (EOD), I have submitted the daily status report to team lead.""",
        "Roles & Responsibilities"
    )

    add_qa(
        "What are your daily activities as an ETL Tester?",
        """And my Daily Activities are:
• Once I login, I go through the outlook if any response required I will reply.
• I involve in stand-up call where team members update ourselves regarding project status like what I did yesterday and what I have to do today and is there any obstacles.
• Then I carry test activities.
• I also involved in scrum calls and client calls.
• At the end of the day (EOD), I have submitted the daily status report to team lead""",
        "Daily Activities"
    )

    add_qa(
        "How do you manage work priority across multiple projects / Priority Management?",
        """How I manage work priority across multiple projects:

• Firstly, I review project timelines and milestone deadlines for each assigned project to understand immediate delivery dates.
• Then I communicate with Project Managers (PMs) and Scrum Masters in the daily standup to align on critical business priorities and current sprint goals.
• I prioritize Sev-1 and Sev-2 blocker defects and critical production pipeline failures first before starting new test case execution for secondary projects.
• If there is any conflicting deadline or bandwidth overlap, I immediately raise it to my Test Lead and Scrum Master to re-estimate or re-align tasks.
• I track all tasks and test executions in Jira/HP ALM and send my Daily Status Report (DSR) at EOD detailing completed items and blockers.""",
        "Project Management & Priorities"
    )

    # Page 2 & 3: Project Architecture / Data Flow
    add_qa(
        "Explain your project architecture / How data flows in your project / Stages of Project",
        """First I speak about my project.
My project name is Menard’s data warehouse client is from United States.
The company sells various numbers of products. Menards has various stores in United States.
They have massive network connection through retailers, stockiest etc.

Stages/Architecture:
1. Source Layer:
In my project there are two types of source. One is flat file and another is data base.
Flat files are generated from client on daily basis and they push it into root folder.
Databases are coming from OLTP system.

2. Landing Layer:
First we bring the data from source to landing layer.
Here we check whether file requirement is met or not.
If requirement meets, flat files are accepted and moved to root folder; if not meets, rejected and archived.

3. Staging Layer:
Where we can cleanse the data (cleansing the data).
We check record count between source and staging.
If requirement not meet then those are sent back to respective sources with error description through email trigger system.

4. Data Warehouse (DWH) Layer:
Now data are loaded in to data warehouse, where we can perform the data validations such as:
• Record count between Source and Target
• Data Validations: Make sure any duplicate and NULL value populated target or not
• Column Mapping using Minus
• SCD type 2 Validations for Initial and Incremental Load
• Standardisation Table and apply business logic. If requirement not meets -> Reject.

5. Data Mart Layer:
Based on Specific Subject oriented, data’s Stored in Datamart Layer.
Top Down approach is used (DWH to Data Marts like HR, Sales, Production, QA).

6. Report Layer:
Finally data is sent to BI reporting tools to generate reports for analysis.""",
        "Project Architecture"
    )

    # Page 4: Defect Life Cycle / Bug Life Cycle
    add_qa(
        "Explain Defect Life Cycle or Bug Life Cycle in ETL Testing",
        """Defect life cycle or Bug life cycle:
1. New / Open: Tester finds a bug and logs it with Open status.
2. Assigned: Defect assigned to Developer.
3. Review / Analysis by Developer:
   - If Valid: Developer investigates further.
   - If Invalid: Developer rejects it (Rejected).
   - If Duplicate: Marked as Duplicate.
   - If Not Reproducible / Cannot be fixed now: Deferred / Hold / Postponed to next release.
4. Fix: If valid, developer fixes the code/mapping and status changes to Fixed / Resolved.
5. Retest: Build deployed to testing environment, Tester performs Retesting.
   - If bug still exists: Reopen.
   - If bug is resolved: Closed.
Status flow: New -> Open -> Assigned -> In Progress -> Fixed -> Retest -> Closed (or Reopen).""",
        "Defect Life Cycle"
    )

    # Page 5: Data Warehouse Concepts
    add_qa(
        "What is a Data Warehouse and what are its characteristics?",
        """Data Warehouse:
It is a repository or place in which we can store the historical data, by which generate the reports and analysis the business key metric fields in order to improve the business.

The Characteristics of data warehouse are:
1. It is subject oriented: in which we can focus on the particular area of analysis.
2. It is Integrated: in which data can be integrated from different sources.
3. It is Time Variant: in which we can store the historical data (5 to 10 years).
4. It is Non-Volatile: in which we cannot perform write operations such as delete and update. Only read operation is performed.""",
        "Data Warehouse"
    )

    # Page 6: Normalization & Denormalization
    add_qa(
        "What is Normalization and Denormalization?",
        """Normalization:
It is the process of splitting the table into another table in order to minimize the redundancy.
• Redundancy data means repeated or duplicate data.
• Mainly used in OLTP systems to reduce data redundancy and improve data integrity.
Forms of Normalization:
- 1NF (First Normal Form): Eliminate repeating groups in individual tables, create separate table for each set of related data, identify each set with a primary key. No multi-valued attributes.
- 2NF (Second Normal Form): Table is in 1NF and all non-key attributes are fully functional dependent on the primary key (no partial dependency).
- 3NF (Third Normal Form): Table is in 2NF and no transitive dependency exists (non-prime attributes do not depend on other non-prime attributes).
- BCNF (Boyce-Codd Normal Form): Higher version of 3NF where every determinant is a candidate key.

Denormalization:
It is the process of combining multiple tables into a single table (adding redundancy) to reduce joins and improve read performance for analytical queries in OLAP/Data Warehouse.""",
        "Data Warehouse"
    )

    # Page 7, 8, 9: Dimension & Fact Tables
    add_qa(
        "What is a Dimension Table and what are the types of Dimension Tables?",
        """Dimension Table:
In data warehouse we store the data in two forms: 1. Dimension table 2. Fact table.
• It contains detail descriptive data about the business.
• It contains textual data (attributes).
• It has a Primary Key / Surrogate Key.

Types of Dimension Tables:
1. Conformed Dimension Table:
The dimension table which is having same meaning to all fact tables.
Ex: Time dimension Table (Dim_time), Date dimension.

2. Junk Dimension Table:
These are the unwanted data’s attribute in a fact table. These are nothing but the collection of transactional code flag which are not same as other dimension table (e.g. Yes/No flags, status codes combined).

3. Role-Playing Dimension Table:
A single dimension table that plays multiple roles in the same fact table with different foreign keys.
Ex: Date dimension playing roles as Order_Date, Ship_Date, Delivery_Date.

4. Slowly Changing Dimension (SCD):
Dimensions where data changes slowly over time (e.g., customer address change).""",
        "Data Warehouse"
    )

    add_qa(
        "What is a Fact Table and what are the types of Fact Tables?",
        """Fact Table:
It is a centralized table in a data warehouse schema that contains business measures/metrics and foreign keys pointing to dimension tables.
• It contains foreign key.
• It contains numeric values (facts/measures).

Types of Fact Table:
1. Additive Fact Table:
In which the fact value is generated by considering all dimension tables (can be summed across all dimensions).
Ex: sales, revenue.

2. Semi-Additive Fact Table:
In which fact value is generated by considering few dimension tables (can be summed across some dimensions, but not all like time).
Ex: quantity in hand (balance quantity), bank balance.

3. Non-Additive Fact Table:
In which fact value is generated without considering any dimension table (cannot be added across any dimension).
Ex: sales tax percentage, unit price, margin ratio.

4. Fact-Less Fact Table:
It does not contain any fact or measure. It contains only foreign keys representing the occurrence of an event.
Ex: Student attendance (Student_ID, Course_ID, Date_ID).""",
        "Data Warehouse"
    )

    add_qa(
        "What is the difference between Dimension Table and Fact Table?",
        """Difference between Dimension Table and Fact Table:

Dimension Table:
• It contains details descriptive about the business
• It contains textual data
• It is having primary key / surrogate key
• In OLTP dimension tables are normalized
• In OLAP dimension tables are de-normalized
• Dimension table contains less data (less records)

Fact Table:
• It contains measures/metrics about the business
• It contains measures in numeric
• It is having foreign key
• It is always de-normalized
• Fact table contains more data (millions of records)""",
        "Data Warehouse"
    )

    # Page 10: Schemas
    add_qa(
        "What is a Schema? Explain Star Schema and Snowflake Schema with differences.",
        """Schema:
Schema is the skeleton structure that represents the logical view of the entire database. It defines how the data is organized and how the relations among them are associated.

Types of Schema:
1. Star Schema:
• A centralized located fact table which surrounded by multiple dimension tables.
• It looks like a star hence it is called star schema.
• Dimension tables are de-normalized.

2. Snowflake Schema:
• It is the extension of star schema, in which dimension tables are exploded/split into another dimension tables (normalized).

Difference between Star Schema and Snowflake Schema:

Star Schema:
• Centralized fact table surrounded by multiple dimension tables.
• Dimension tables are de-normalized.
• Simple in structure hence joins and queries are simple.
• Execution of SQL query is high performance (faster).
• Consumes more storage due to redundancy.

Snowflake Schema:
• Extension of star schema where dimension tables are exploded into sub-dimension tables.
• Dimension tables are normalized.
• Complex in structure hence joins and tables are complicated.
• Performance of the query is less (slower due to multiple joins).
• Consumes less storage space (less redundancy).""",
        "Data Warehouse"
    )

    # Page 11 - 20: SQL Sub-languages and Commands
    add_qa(
        "What is SQL and what are the sub-languages / command categories in SQL?",
        """SQL (STRUCTURED QUERY LANGUAGE):
• SQL is a standard query language for storing, manipulating and retrieving data in databases.
• It is a command based language.
• Each command having its own meaning.
• By this language we can communicate with DB or DWH.
• It is non-case sensitive.

Commands / Sub-languages of SQL:
1. DDL (Data Definition Language):
   Create, Alter (Add, Modify, Drop, Rename), Drop, Rename, Purge, Flashback, Truncate.
2. DML (Data Manipulation Language):
   Insert, Update, Delete.
3. DQL (Data Query Language):
   Select.
4. DCL (Data Control Language):
   Grant, Revoke.
5. TCL (Transaction Control Language):
   Commit, Rollback, Savepoint.

Data Types in SQL:
• Number: It accepts only numerical value.
• Char: Fixed length character string (size 0-255). Unused space is padded with empty spaces.
• Varchar2: Flexible/Variable length character string (size 0-4000). Unused space is returned to DB.
• Date: Accepts date and time values.""",
        "SQL"
    )

    add_qa(
        "Explain DDL commands with syntax and examples (CREATE, ALTER, DROP, RENAME, PURGE, FLASHBACK, TRUNCATE).",
        """DDL (Data Definition Language):
DDL commands are used to define the DB objects. They directly interact with DB hence auto commit (automatic save) and cannot be rolled back. They deal only with the structure of the table.

1. CREATE: Define/create DB objects like tables, views, procedures.
Syntax: CREATE TABLE <TABLE_NAME> (COL1 DATATYPE(SIZE), COL2 DATATYPE(SIZE)...);
Example: CREATE TABLE COLLEGE (REGNUMBER VARCHAR2(10), STUDENTNAME CHAR(30), PHONENUMBER NUMBER(10), BRANCH VARCHAR2(10));

2. DESC: Display table structure.
Example: DESC COLLEGE;

3. ALTER: Modify table structure.
• ADD: ALTER TABLE COLLEGE ADD (HOD VARCHAR2(10), CITY VARCHAR2(20));
• MODIFY: ALTER TABLE COLLEGE MODIFY (HOD CHAR(10));
• DROP COLUMN: ALTER TABLE COLLEGE DROP (CITY);
• RENAME COLUMN: ALTER TABLE COLLEGE RENAME COLUMN HOD TO BRANCHHEAD;

4. DROP: Permanently delete table from DB.
Syntax: DROP TABLE COLLEGE;

5. RENAME: Rename table.
Syntax: RENAME COLLEGE TO COLLEGE_1;

6. PURGE: Remove DB objects permanently from recycle bin.
Syntax: PURGE TABLE COLLAGE; or DROP TABLE COLLAGE PURGE;

7. FLASHBACK: Restore DB objects from recycle bin.
Syntax: FLASHBACK TABLE COLLAGE TO BEFORE DROP;
View recycle bin: SELECT * FROM RECYCLEBIN; or SHOW RECYCLEBIN;

8. TRUNCATE: Delete all data in a single shot while structure remains same.
Syntax: TRUNCATE TABLE COLLEGE_1;""",
        "SQL"
    )

    add_qa(
        "What is the difference between DROP, TRUNCATE, and DELETE?",
        """Differences between DROP, TRUNCATE, and DELETE:

DROP:
• DDL command.
• Deletes both data and structure of table permanently from DB in a single shot.
• Auto commit (No rollback).
• Performance is very high.

TRUNCATE:
• DDL command.
• Deletes all records from the table in a single shot, but structure remains intact.
• Auto commit (No rollback).
• Performance is high (resets high water mark, doesn't log individual row deletions).
• Cannot use WHERE clause.

DELETE:
• DML command.
• Deletes specific row(s) or all rows from table based on condition.
• Not auto commit (can be rolled back before commit).
• Performance is slower for large tables because it logs each deleted row in undo/redo logs.
• Can use WHERE clause.""",
        "SQL"
    )

    add_qa(
        "Explain DML commands with syntax and examples (INSERT, UPDATE, DELETE).",
        """DML (Data Manipulation Language):
DML commands interact with DB through buffer, hence they are NOT auto commit (can be rolled back). Used to manipulate or organize data.

1. INSERT:
a. All columns: INSERT INTO COLLAGE_1 VALUES ('1', 'RAVI', 'ME');
b. Specific columns: INSERT INTO COLLAGE_1 (SLNO, NAME, BRANCH) VALUES ('1', 'RAVI', 'ME');
c. Multiple values prompt: INSERT INTO COLLAGE_1 VALUES ('&SLNO', '&NAME', '&BRANCH');

2. UPDATE:
a. All rows: UPDATE COLLAGE_1 SET BRANCH = 'COMP SCI';
b. With WHERE clause: UPDATE COLLAGE_1 SET BRANCH = 'COMP SCI' WHERE NAME = 'RAVI';

3. DELETE:
a. Specific rows: DELETE FROM COLLAGE_1 WHERE SLNO = 4;
b. All rows: DELETE FROM COLLAGE_1;""",
        "SQL"
    )

    add_qa(
        "What are DCL and TCL commands in SQL?",
        """DCL (Data Control Language):
Used to control access and permissions to database objects.
1. GRANT: Gives user access privileges to database.
   Syntax: GRANT SELECT, INSERT ON EMP TO USER1;
2. REVOKE: Withdraws user access privileges.
   Syntax: REVOKE INSERT ON EMP FROM USER1;

TCL (Transaction Control Language):
Used to manage transactions in the database.
1. COMMIT: Saves all transactions permanently to DB.
   Syntax: COMMIT;
2. ROLLBACK: Undoes transactions that have not been committed.
   Syntax: ROLLBACK; / ROLLBACK TO SAVEPOINT SP1;
3. SAVEPOINT: Creates a checkpoint within a transaction to rollback partially.
   Syntax: SAVEPOINT SP1;""",
        "SQL"
    )

    # Page 21 - 22: NULL Handling Functions
    add_qa(
        "Explain NVL, NVL2, NULLIF, and COALESCE functions in SQL with examples.",
        """NULL Handling Functions in SQL:

1. NVL: Replaces NULL value with a replacement value.
Syntax: NVL(expression, replacement_value)
Example: SELECT EMP.*, NVL(COMM, 0) FROM EMP;
Note: Both arguments must be of compatible datatype. If COMM is null, returns 0.

2. NVL2: Replaces based on whether expression is NOT NULL or NULL.
Syntax: NVL2(expression, value_if_not_null, value_if_null)
Example: SELECT EMP.*, NVL2(COMM, '5000', '2000') FROM EMP;
Passes 3 parameters.

Difference NVL vs NVL2:
• NVL replaces only NULL values (2 parameters).
• NVL2 replaces both NOT NULL and NULL values (3 parameters).

3. NULLIF: Compares two expressions. If they are equal, returns NULL; if not equal, returns the first expression.
Syntax: NULLIF(expr1, expr2)
Example: SELECT NULLIF(10, 10) FROM DUAL; -> NULL
Example: SELECT NULLIF(10, 20) FROM DUAL; -> 10

4. COALESCE: Returns the first non-null expression from the list of arguments.
Syntax: COALESCE(val1, val2, val3, ... val_n)
Examples:
• SELECT COALESCE(NULL, 1, 2, 3, 4) FROM DUAL; -> Returns 1
• SELECT COALESCE(NULL, NULL, 2, 3, 4) FROM DUAL; -> Returns 2
• SELECT COALESCE(NULL, NULL, NULL) FROM DUAL; -> Returns NULL
• SELECT EMP.*, COALESCE(COMM, SAL, DEPTNO) FROM EMP;""",
        "SQL Functions"
    )

    # Page 23 - 26: Character Functions
    add_qa(
        "Explain Character functions in SQL (UPPER, LOWER, INITCAP, CONCAT, LENGTH, SUBSTR, INSTR, REPLACE, TRANSLATE).",
        """SQL Character Functions:
Accept character inputs and return character or numeric values.

1. Case-Manipulative Functions:
• UPPER('geeks') -> 'GEEKS'
• LOWER('GEEKS') -> 'geeks'
• INITCAP('hello world') -> 'Hello World'

2. Character-Manipulative Functions:
• CONCAT('computer', 'science') -> 'computerscience'
• LENGTH('Database') -> 8
• SUBSTR: Extracts part of string.
  Syntax: SUBSTR(string, start_position, [length])
  - SELECT SUBSTR('Database Management', 10, 6) FROM DUAL; -> 'Manage'
  - SELECT SUBSTR('RAJASHEKHAR', 4, 6) FROM DUAL; -> 'ASHEKH'
  - SELECT SUBSTR('RAJASHEKHAR', -5, 2) FROM DUAL; -> 'EK'
• INSTR: Returns the position of a substring within a string.
  Syntax: INSTR(string, search_substring, [start_pos], [nth_occurrence])
  - SELECT INSTR('Google apps are great applications', 'app', 1, 2) FROM DUAL; -> 23
  - SELECT INSTR('rajashekhar', 'a', 1, 3) FROM DUAL; -> 10
• REPLACE: Replaces occurrence of a string with another string.
  Syntax: REPLACE(text, search_str, replacement_str)
  - SELECT REPLACE('DATA MANAGEMENT', 'DATA', 'DATABASE') FROM DUAL; -> 'DATABASE MANAGEMENT'
• TRANSLATE: Translates one-to-one character by character.
  Syntax: TRANSLATE(string, from_chars, to_chars)
  - SELECT TRANSLATE('abcdef', 'abc', 'bcd') FROM DUAL; -> 'bcddef'

Difference REPLACE vs TRANSLATE:
• REPLACE replaces whole string / substring matching.
• TRANSLATE substitutes individual characters one-by-one based on position.""",
        "SQL Functions"
    )

    add_qa(
        "How do you extract username, domain name, and extension from an email in SQL using SUBSTR and INSTR?",
        """Extracting parts of an email (e.g., 'rajashekhar@gmail.com'):

1. Extract extension (.com):
SELECT SUBSTR('rajashekhar@gmail.com', INSTR('rajashekhar@gmail.com', '.')) FROM DUAL;
-- Output: .com
SELECT SUBSTR('rajashekhar@gmail.com', INSTR('rajashekhar@gmail.com', '.') + 1) FROM DUAL;
-- Output: com

2. Extract domain name (gmail):
SELECT SUBSTR('rajashekhar@gmail.com',
  INSTR('rajashekhar@gmail.com', '@') + 1,
  INSTR('rajashekhar@gmail.com', '.') - INSTR('rajashekhar@gmail.com', '@') - 1)
FROM DUAL;
-- Output: gmail

3. Extract username (rajashekhar):
SELECT SUBSTR('rajashekhar@gmail.com', 1, INSTR('rajashekhar@gmail.com', '@') - 1) FROM DUAL;
-- Output: rajashekhar""",
        "SQL Queries"
    )

    # Page 27 - 30: Date Functions & DECODE
    add_qa(
        "Explain Date Functions in Oracle SQL (SYSDATE, TO_CHAR, MONTHS_BETWEEN, ADD_MONTHS, NEXT_DAY, LAST_DAY).",
        """Date Functions in Oracle SQL:
Oracle stores dates in internal numeric format representing century, year, month, day, hours, minutes, seconds. Default format: DD-MON-YY.

1. SYSDATE: Returns current system date and time.
   SELECT SYSDATE FROM DUAL;

2. TO_CHAR: Formats date to string.
   SELECT TO_CHAR(SYSDATE, 'DD-MON-YYYY') FROM DUAL;
   SELECT TO_CHAR(HIREDATE, 'DD.MM.YYYY:HH24:MI:SS') FROM EMP;
   SELECT * FROM EMP WHERE TO_CHAR(HIREDATE, 'YYYY') = '1981';

3. MONTHS_BETWEEN: Calculates number of months between date1 and date2.
   Syntax: MONTHS_BETWEEN(date1, date2)
   SELECT MONTHS_BETWEEN('31-MAR-1995', '28-FEB-1994') FROM DUAL; -> 13
   SELECT MONTHS_BETWEEN(SYSDATE, HIREDATE)/12 AS EXP_YEARS FROM EMP;

4. ADD_MONTHS(date, n): Adds n months to date.
5. LAST_DAY(date): Returns last day of the month for given date.
6. NEXT_DAY(date, 'DAY'): Returns date of the next specified day.""",
        "SQL Functions"
    )

    add_qa(
        "Explain DECODE function and CASE statement in SQL with examples.",
        """DECODE and CASE Statement in SQL:

1. DECODE Function:
Acts as IF-THEN-ELSE statement in Oracle SQL.
Syntax: DECODE(column/expr, search1, result1, search2, result2, ..., default_result)
Example:
SELECT supplier_name,
       DECODE(supplier_id, 10000, 'IBM',
                           10001, 'Microsoft',
                           10002, 'Hewlett Packard',
                           'Gateway') AS result
FROM suppliers;

2. CASE Statement:
Standard SQL conditional statement.
Syntax:
CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ELSE default_result
END

Example:
SELECT first_name, salary,
  CASE
    WHEN salary > 10000 THEN 'High'
    WHEN salary BETWEEN 5000 AND 10000 THEN 'Medium'
    ELSE 'Low'
  END AS salary_grade
FROM employees;

Difference DECODE vs CASE:
• DECODE is Oracle-specific; CASE is ANSI SQL standard (works in all databases).
• DECODE does equality comparison only; CASE supports complex expressions, logical operators (<, >, BETWEEN, IN, AND, OR).""",
        "SQL Functions"
    )

    # Page 33 - 34: Analytical Functions
    add_qa(
        "Explain Analytical / Window Functions in SQL (ROW_NUMBER, RANK, DENSE_RANK) and their differences.",
        """Analytical / Window Functions (ROW_NUMBER vs RANK vs DENSE_RANK):

1. ROW_NUMBER(): Assigns a unique sequential integer to each row starting from 1, regardless of duplicate values.
2. RANK(): Assigns rank with gaps. If there are duplicates, they get the same rank, and next rank is skipped.
3. DENSE_RANK(): Assigns rank without gaps. If duplicates exist, they get the same rank, and next rank continues consecutively.

Example on Salaries: [5000, 4000, 4000, 3000]
- ROW_NUMBER: 1, 2, 3, 4
- RANK:       1, 2, 2, 4 (skips 3)
- DENSE_RANK: 1, 2, 2, 3 (no skip)

Syntax:
SELECT first_name, salary, department_id,
  ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as row_num,
  RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as rnk,
  DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as drank
FROM employees;

To find 2nd highest salary:
SELECT * FROM (
  SELECT employees.*, DENSE_RANK() OVER (ORDER BY salary DESC) as drank
  FROM employees
) WHERE drank = 2;""",
        "SQL Analytical Functions"
    )

    # Page 35 - 36: Joins
    add_qa(
        "Explain different types of Joins in SQL (Inner, Left Outer, Right Outer, Full Outer, Cross, Self Join).",
        """Types of Joins in SQL:
Used to retrieve data from two or more tables based on a related column.

1. Inner Join: Returns only matching records from both tables.
   SELECT * FROM EMP E INNER JOIN DEPT D ON E.DEPTNO = D.DEPTNO;

2. Left Outer Join: Returns all records from left table and matched records from right table; non-matching columns from right table will be NULL.
   SELECT * FROM EMP E LEFT JOIN DEPT D ON E.DEPTNO = D.DEPTNO;

3. Right Outer Join: Returns all records from right table and matched records from left table; non-matching columns from left table will be NULL.
   SELECT * FROM EMP E RIGHT JOIN DEPT D ON E.DEPTNO = D.DEPTNO;

4. Full Outer Join: Returns all records when there is a match in either left or right table. Unmatched records on either side have NULL values.
   SELECT * FROM EMP E FULL OUTER JOIN DEPT D ON E.DEPTNO = D.DEPTNO;

5. Cross Join (Cartesian Product): Returns Cartesian product of records (m * n rows).
   SELECT * FROM EMP CROSS JOIN DEPT;

6. Self Join: A table joined with itself to compare rows within the same table (e.g. employee and manager relationship).
   SELECT E.ENAME AS Emp, M.ENAME AS Manager FROM EMP E JOIN EMP M ON E.MGR = M.EMPNO;""",
        "SQL Joins"
    )

    # Page 37 - 39: Agile Methodology & Scrum
    add_qa(
        "Explain Agile Methodology, Scrum Framework, Roles, Artifacts, and Ceremonies.",
        """Agile Methodology:
Agile is an iterative and incremental approach to software development and testing where requirements and solutions evolve through collaboration.

Scrum Roles:
1. Product Owner (PO): Defines requirements, maintains product backlog, prioritizes user stories based on business value.
2. Scrum Master (SM): Facilitates Scrum ceremonies, removes impediments/blockers, ensures team adheres to Scrum practices.
3. Scrum Team / Development & QA Team: Cross-functional team responsible for delivering potentially shippable increment.

Scrum Artifacts:
1. Product Backlog: Master list of all user stories, features, and fixes required for the product.
2. Sprint Backlog: Subset of product backlog items selected for execution in current sprint.
3. Product Increment: Working software delivered at the end of sprint.

Scrum Ceremonies / Meetings:
1. Sprint Planning: Team discusses and commits to backlog items for upcoming sprint (typically 2-3 weeks).
2. Daily Standup (Daily Scrum): 15-minute daily meeting answering 3 questions: What did I do yesterday? What will I do today? Any blockers/obstacles?
3. Sprint Review: Demo of completed increment to stakeholders.
4. Sprint Retrospective: Team reviews what went well, what didn't go well, and improvement actions for next sprint.
5. Backlog Grooming / Refinement: Reviewing and estimating user stories for upcoming sprints.""",
        "Agile Methodology"
    )

    # Page 40: Levels of Testing
    add_qa(
        "What are the levels of testing?",
        """Levels of Testing:
1. Unit Testing: Done by developers to test individual modules/components/code units.
2. Integration Testing: Testing data flow and interaction between integrated modules/interfaces.
3. System Testing: End-to-end testing of the complete integrated application against requirements.
4. User Acceptance Testing (UAT): Tested by client/end-users in UAT environment before production deployment (Alpha and Beta testing).""",
        "Testing Basics"
    )

    # Page 41 - 50: ETL Test Scenarios, Record Count, Minus Queries, SCD Testing
    add_qa(
        "How do you perform Record Count Validation in ETL Testing?",
        """Record Count Validation between Source and Target:
Test Case Scenario: Compare total record counts between Source table/files and Target table.

SQL Queries:
1. Source Count:
   SELECT COUNT(*) FROM SOURCE_TABLE;
2. Target Count:
   SELECT COUNT(*) FROM TARGET_TABLE;
3. If filters/transformation applied:
   SELECT COUNT(*) FROM SOURCE_TABLE WHERE <FILTER_CONDITION>;
   SELECT COUNT(*) FROM TARGET_TABLE;
Both counts must match. If difference exists, investigate rejected records, filter dropouts, or duplicates.""",
        "ETL Testing"
    )

    add_qa(
        "How do you perform Column Mapping and Data Validation using MINUS query in ETL Testing?",
        """Data Validation using MINUS query:
Validates that all data in source is correctly transformed and loaded into target without corruption or mismatch.

Approach:
1. Source minus Target:
   SELECT COL1, COL2, COL3 FROM SOURCE_TABLE
   MINUS
   SELECT COL1, COL2, COL3 FROM TARGET_TABLE;
   -- Expected: 0 rows (No records in source missing in target).

2. Target minus Source:
   SELECT COL1, COL2, COL3 FROM TARGET_TABLE
   MINUS
   SELECT COL1, COL2, COL3 FROM SOURCE_TABLE;
   -- Expected: 0 rows (No extra or incorrect records in target).

If MINUS query returns rows, investigate data transformation mismatch, truncation, padding, or missing records.""",
        "ETL Testing"
    )

    add_qa(
        "What are the test cases and validation steps for SCD Type 2 (Initial Load and Incremental Load)?",
        """SCD Type-2 (Slowly Changing Dimension Type 2) Validation:
SCD Type 2 maintains full history by adding a new record with versioning/effective dates whenever a dimension attribute changes.

Generic fields generated by ETL:
• Surrogate Key (SK)
• Natural Key (ID)
• ETL Effective Start Date (e.g. 01-Jan-20)
• ETL Effective End Date (e.g. 31-Dec-2099 for active row)
• Active Row Flag ('A' for active, 'H' for historical / 'Y'/'N')
• Version Number (1, 2, 3...)
• ETL Process Date

Validation Test Cases:
1. Initial Load:
   - Verify job runs successfully.
   - Surrogate Key is generated automatically, in sequential incremental numeric order.
   - ETL Effective Start Date and ETL Process Date must be same (current batch date).
   - ETL Effective End Date must be high future date (e.g., 31-Dec-2099 or 9999-12-31).
   - Active Row Flag must be 'A' (Active).
   - Version Number must be 1.

2. Incremental / Delta Load (when a record changes, e.g. City changed from New York to Florida):
   - For Old Record: Active Row Flag changes from 'A' to 'H' (Historical). Effective End Date updates to (New Batch Date - 1).
   - For New Record: Inserted with new Surrogate Key (incremental). Version number increments to 2. Active Row Flag set to 'A'. Effective Start Date is new batch date. Effective End Date set to 31-Dec-2099.
   - Unchanged records remain intact with Version 1 and Active flag 'A'.""",
        "SCD"
    )

    # Page 51: Primary Key vs Foreign Key vs Surrogate Key
    add_qa(
        "What is the difference between Primary Key and Surrogate Key?",
        """Difference between Primary Key and Surrogate Key:

Primary Key:
• Used for maintaining unique records in OLTP database.
• Can be alphanumeric or numeric.
• It is a business attribute / natural key from source.
• It is actual table/business data.
• Given by user / source system.

Surrogate Key:
• Used to maintain unique records in OLAP database / Data Warehouse.
• Always a system-generated sequential numeric number.
• It is a technical attribute created by ETL pipeline.
• Not part of source business data.
• Generated automatically by ETL/database sequence (e.g., in dimension tables for SCD tracking).""",
        "Data Warehouse"
    )

    add_qa(
        "What is the difference between Primary Key, Foreign Key, and Surrogate Key?",
        """Difference between Primary Key, Foreign Key, and Surrogate Key:

1. Primary Key (PK):
• Uniquely identifies each record in an OLTP table.
• Does not allow NULL or duplicate values.
• It is a natural business key originating from source systems.
• Can be alphanumeric or numeric.

2. Foreign Key (FK):
• Creates a relationship/referential integrity between two tables (links child table to parent table PK).
• In Data Warehouse, Fact tables contain Foreign Keys that reference Primary/Surrogate Keys of Dimension tables.
• Allows duplicate and NULL values (unless NOT NULL is specified).
• Business or relational attribute.

3. Surrogate Key (SK):
• Unique numeric identifier generated artificially by ETL pipeline for OLAP / Data Warehouse dimension tables.
• It has no business meaning (pure technical attribute).
• Always sequential numeric integer (1, 2, 3...).
• Used in Dimension tables to track history (e.g. SCD Type 2 where natural PK repeats for new versions).""",
        "Data Warehouse"
    )

    add_qa(
        "What is a Foreign Key and how is it used in Data Warehouse / ETL?",
        """Foreign Key in ETL and Data Warehouse:
• A column or set of columns in a table that refers to the Primary Key in another table.
• Used to establish parent-child relationship and maintain referential integrity.
• In Data Warehouse dimensional modeling:
  - Fact table contains Foreign Keys pointing to the Surrogate Keys of Dimension tables.
  - Dimension table contains Surrogate Key / Primary Key.
• ETL Validation Check: Ensure no orphan records in Fact tables (every FK in Fact must match an existing SK in Dimension table).""",
        "Data Warehouse"
    )

    # Page 52 - 55: Mismatch records, ETL Bugs, OLTP vs OLAP, SUBQUERY vs CORRELATED SUBQUERY
    add_qa(
        "If you find mismatch records during ETL testing, what is your approach?",
        """Approach when mismatch records are found:
1. Extract and isolate the mismatch records along with screenshots/log artifacts.
2. Cross-verify the SQL query, STM (Source Target Mapping) document, and transformation rules.
3. Check reject files / bad files / session logs in Informatica for transformation errors.
4. Contact the on-site coordinator and dev team to discuss whether it is a data issue, mapping gap, or ETL bug.
5. Log a defect in HP ALM / Jira with proper severity, steps to reproduce, source & target queries, and query output attachments.
6. Follow up in stand-up call and defect review meeting until resolved, then retest upon deployment.""",
        "ETL Testing"
    )

    add_qa(
        "What is ETL testing and what are common types of bugs in ETL?",
        """ETL Testing:
ETL stands for Extraction, Transformation, and Loading.
• Extraction: Extracting data from heterogeneous sources (flat files, databases, APIs).
• Transformation: Applying conversion, cleansing, aggregation, and business rules so data is suitable for analytical reporting.
• Loading: Loading processed data into target Data Warehouse/Data Marts.

Common Types of Bugs in ETL:
1. Source Bugs: Missing source data, corrupt delimiter, format mismatch, dirty data.
2. Load Condition Bugs: Truncation, primary key / foreign key violation, rejected rows.
3. Calculation / Transformation Bugs: Incorrect formula, wrong date format, null handling logic failure.
4. User Interface / Reporting Bugs: BI tool display mismatches, wrong aggregation in reports.
5. Duplicate Data Bugs: Duplicate records loaded into target due to missing distinct/grouping.
6. Performance Bugs: Long running queries, session timeout, bottleneck in transformation.""",
        "ETL Testing"
    )

    add_qa(
        "What is the difference between OLTP and OLAP? / Difference between Database and Data Warehouse / OLTP vs OLAP",
        """Difference between OLTP (Database) and OLAP (Data Warehouse):

OLTP (Database / Online Transaction Processing):
• Records user current transaction data.
• Tables and joins are complex because they are normalized (3NF) to eliminate duplicates.
• Optimized for write operations (INSERT, UPDATE, DELETE).
• Kept for small to medium data volume (MB to GB).
• Application/Transaction oriented.
• Volatile data, handling single record at a time.
• Uses Entity-Relationship (ER) modeling.

OLAP (Data Warehouse / Online Analytical Processing):
• Maintains historical business data for analytics and decision making.
• Tables and joins are simple because they are de-normalized (Star/Snowflake).
• Optimized for read-only operations and complex aggregation queries.
• Stores large to very large data volume (GB to TB to PB).
• Subject oriented.
• Non-volatile (read-mostly), handling millions of records at a time.
• Uses Dimensional modeling (Facts and Dimensions).""",
        "Data Warehouse"
    )

    add_qa(
        "What is the difference between ER Modeling and Dimensional Modeling? / ER vs Dimensional Modeling / Entity Relationship vs Dimensional",
        """Difference between Entity-Relationship (ER) Modeling and Dimensional Modeling:

1. Entity-Relationship (ER) Modeling (OLTP):
• Purpose: Designed for operational transaction processing (OLTP).
• Normalization: Highly normalized (3NF / Third Normal Form) to eliminate data redundancy and anomalies.
• Structure: Uses Entities, Attributes, and Relationships (Primary Key / Foreign Key links).
• Optimization: Optimized for write operations (INSERT, UPDATE, DELETE).
• Data Volume: Typically holds detailed current/live transaction data.
• Modeling Tools: Erwin Data Modeler, ER/Studio, Microsoft Visio.

2. Dimensional Modeling (OLAP / DWH):
• Purpose: Designed for data analysis, BI reporting, and decision making (OLAP).
• Normalization: De-normalized into Star Schema or Snowflake Schema for query performance.
• Structure: Centered around Fact Tables (numeric metrics/measures) surrounded by Dimension Tables (descriptive textual attributes).
• Optimization: Optimized for read-only aggregations and complex reporting queries.
• Data Volume: Holds vast historical business data across multiple years.
• Modeling Architecture: Star Schema, Snowflake Schema, Galaxy / Fact Constellation Schema.

Validation in ETL Testing:
• I verify that when data is extracted from the 3NF ER source database and loaded into the Dimensional Data Warehouse, foreign keys link correctly to dimension surrogate keys, duplicate measures are eliminated, and historical tracking (SCD Type 2) maintains active/historical flags.""",
        "Data Warehouse"
    )

    add_qa(
        "What is the difference between Nested Subquery and Correlated Subquery?",
        """Difference between Nested Subquery and Correlated Subquery:

Nested / Inner Subquery:
• Inner query and outer query are independent.
• Inner query executes first, only once, and passes result to outer query.
• High performance.
• Example: SELECT * FROM EMP WHERE SAL > (SELECT AVG(SAL) FROM EMP);

Correlated Subquery:
• Inner query and outer query are interdependent (inner query references outer query columns).
• Inner query executes repeatedly once for every row processed by outer query.
• Low performance for large datasets.
• Example: SELECT E.ENAME, E.SAL, E.DEPTNO FROM EMP E WHERE E.SAL > (SELECT AVG(I.SAL) FROM EMP I WHERE I.DEPTNO = E.DEPTNO);""",
        "SQL"
    )

    add_qa(
        "What is the difference between IN and EXISTS in SQL?",
        """Difference between IN and EXISTS:

IN:
• Evaluates true if value exists in list/subquery.
• Executes inner subquery first, loads entire result into memory, then matches outer query.
• Slower performance when subquery returns large result set.
• Does not handle NULL values properly in NOT IN.

EXISTS:
• Evaluates true as soon as it finds the first matching row in subquery (boolean check).
• Terminates subquery execution upon first match without scanning entire table.
• Higher performance when subquery has large amount of data or is correlated.
• Handles NULLs gracefully.""",
        "SQL"
    )

    add_qa(
        "What is a View and what is the difference between View and Materialized View?",
        """View vs Materialized View:

View (Simple / Complex View):
• Virtual table based on result of a SQL query.
• Does not store data physically; stores only query definition.
• Every time view is queried, underlying query is executed against base tables.
• Always reflects real-time data.

Materialized View:
• Physical copy of query results stored on disk like a table.
• Stores data physically and occupies storage space.
• Querying materialized view is fast because data is precomputed.
• Needs refresh mechanism (FAST, COMPLETE, ON COMMIT, ON DEMAND) to sync with base table changes.
• Widely used in Data Warehouses for heavy aggregation reports.""",
        "SQL"
    )

    # Page 59 - 67: Unix Commands
    add_qa(
        "What are essential Unix commands used in ETL testing? (head, tail, grep, sed, awk, find, cut, sort, uniq, wc, chmod)",
        """Essential Unix Commands in ETL Testing:

1. File Viewing & Navigation:
• head -n 10 file.txt: View first 10 lines of file.
• tail -n 10 file.txt: View last 10 lines of file.
• tail -f logfile.log: Monitor log file in real-time.
• cat file.txt: Display entire file content.
• more / less file.txt: Page-by-page file viewing.

2. File & Line Count:
• wc -l file.txt: Count total lines in file (used for source file record count validation).
• wc -w file.txt: Count words; wc -c file.txt: Count bytes.

3. Search & Pattern Matching (grep):
• grep 'PATTERN' file.txt: Search for string in file.
• grep -i 'pattern' file.txt: Case insensitive search.
• grep -v 'pattern' file.txt: Invert match (show lines NOT containing pattern).
• grep -c 'pattern' file.txt: Count matching lines.

4. Text Processing (sed, awk, cut):
• cut -d',' -f1,3 file.csv: Extract columns 1 and 3 with delimiter comma.
• sed 's/old/new/g' file.txt: Replace all occurrences of old with new.
• awk -F',' '{print $1, $3}' file.csv: Print fields 1 and 3.
• awk -F',' '$3 > 1000 {print $1, $3}' file.csv: Filter records where column 3 > 1000.

5. Sorting & Deduplication (sort, uniq):
• sort file.txt: Sort lines alphabetically.
• sort -n file.txt: Numerical sort.
• sort file.txt | uniq: Remove adjacent duplicate lines.
• sort file.txt | uniq -c: Count duplicate occurrences.
• sort file.txt | uniq -d: Display only duplicate lines.

6. File Search & Permissions:
• find /path -name '*.csv' -print: Find files by name pattern.
• chmod 755 script.sh: Change file permissions (rwxr-xr-x).
• rm -rf dir_name: Remove directory and contents.
• cmp / diff file1 file2: Compare two files line-by-line.""",
        "Unix Commands"
    )

    # Page 68 - 76: Complex SQL Queries & Interview Scenarios
    add_qa(
        "Write SQL queries to find Nth highest salary in Oracle/SQL (different approaches).",
        """SQL queries to find Nth highest salary:

1. Approach 1: Using DENSE_RANK() (Recommended & Standard)
SELECT * FROM (
    SELECT first_name, salary, department_id,
           DENSE_RANK() OVER (ORDER BY salary DESC) as drank
    FROM employees
) WHERE drank = 2; -- Change to N for Nth highest salary

2. Approach 2: Using Correlated Subquery
SELECT * FROM employees E1
WHERE (N - 1) = (
    SELECT COUNT(DISTINCT E2.salary)
    FROM employees E2
    WHERE E2.salary > E1.salary
);
-- For 2nd highest:
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

3. Approach 3: Department-wise 2nd Highest Salary
SELECT * FROM (
    SELECT e.first_name, e.salary, e.department_id, d.department_name,
           DENSE_RANK() OVER (PARTITION BY e.department_id ORDER BY e.salary DESC) as drank
    FROM employees e
    INNER JOIN departments d ON e.department_id = d.department_id
) WHERE drank = 2;""",
        "SQL Queries"
    )

    add_qa(
        "Write SQL query to find and delete duplicate records from a table.",
        """Find and Delete Duplicate Records in SQL:

1. Identify Duplicate Records:
SELECT col1, col2, COUNT(*)
FROM table_name
GROUP BY col1, col2
HAVING COUNT(*) > 1;

2. Delete Duplicate Records using ROWID:
DELETE FROM table_name
WHERE ROWID NOT IN (
    SELECT MIN(ROWID)
    FROM table_name
    GROUP BY col1, col2
);

3. Delete Duplicates using ROW_NUMBER():
DELETE FROM (
    SELECT ROW_NUMBER() OVER (PARTITION BY col1, col2 ORDER BY col1) as rn
    FROM table_name
) WHERE rn > 1;""",
        "SQL Queries"
    )

    add_qa(
        "Write SQL queries to find current experience of employees and employees hired in a specific year.",
        """Employee Experience and Date Queries in Oracle SQL:

1. Current Experience of Employees (in days, months, years):
SELECT first_name, last_name, hire_date,
       ROUND(SYSDATE - hire_date, 2) AS exp_days,
       ROUND(MONTHS_BETWEEN(SYSDATE, hire_date), 2) AS exp_months,
       ROUND(MONTHS_BETWEEN(SYSDATE, hire_date)/12, 2) AS exp_years
FROM employees;

2. Employees hired in year 1981:
SELECT * FROM emp
WHERE TO_CHAR(hiredate, 'YYYY') = '1981';
-- OR
SELECT * FROM emp
WHERE hiredate BETWEEN '01-JAN-1981' AND '31-DEC-1981';""",
        "SQL Queries"
    )

    add_qa(
        "Write SQL query to display employees whose salary is greater than the average salary or greater than department average salary.",
        """Salary comparison queries:

1. Employees with salary greater than overall average salary:
SELECT first_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

2. Employees with salary greater than their department's average salary:
SELECT e.first_name, e.salary, e.department_id, dept_avg.avg_sal
FROM employees e
JOIN (
    SELECT department_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department_id
) dept_avg ON e.department_id = dept_avg.department_id
WHERE e.salary > dept_avg.avg_sal;""",
        "SQL Queries"
    )

    # Page 77 - 81: Challenges Faced, Tasks, Scenarios, and Checklists
    add_qa(
        "What are the challenges faced during ETL testing?",
        """Challenges faced during ETL testing:
• Data loss during the ETL process.
• Incorrect, incomplete, or duplicate data coming from source systems.
• DW system contains historical data, so the data volume is too large (millions/billions of records) and extremely complex to validate in the target system.
• ETL testers are normally not provided with access to see job schedules in the ETL tool, and hardly have access to BI Reporting tools to see final layout of reports.
• Tough to generate and build test cases as data volume is too high and complex.
• ETL testers normally don’t have an idea of end-user report requirements and business flow of the information.
• ETL testing involves various complex SQL queries and analytical functions for data validation.
• Sometimes testers are not provided with up-to-date Source-to-Target Mapping (STM) documents or change logs.""",
        "ETL Challenges"
    )

    add_qa(
        "What are the tasks to be performed by an ETL Tester across Source, Transformation, and Load phases?",
        """Tasks to be performed by an ETL Tester:

1. Verify Tables in the Source System:
• Count check
• Reconcile records with source data
• Data type check
• Ensure no spam / corrupted data loaded
• Remove / detect duplicate data
• Check all keys are in place

2. Apply Transformation Logic:
• Data threshold validation check (e.g. age shouldn't be > 100)
• Record count check before and after transformation logic applied
• Data flow validation from staging area to intermediate tables
• Surrogate key check

3. Data Loading:
• Record count check from intermediate table to target system
• Ensure key field data is not missing or NULL
• Check aggregate values and calculated measures are loaded properly in fact tables
• Check dimensional modeling integrity (referential integrity between Facts and Dimensions)""",
        "ETL Testing"
    )

    add_qa(
        "What are the core ETL Test Scenarios and Validations performed in an ETL Testing Process?",
        """ETL Test Scenarios and Validations:

1. Structure Validation:
• Validate source and target table structures against STM (Source-to-Target Mapping).
• Validate datatypes, length, and precision in source and target.
• Validate column names and order in target system.

2. Mapping Document Validation:
• Validate mapping document has complete change log, datatypes, length, and transformation business rules.

3. Constraint Validation:
• Validate Primary Key, Foreign Key, Unique, Not Null, and Check constraints on target tables.
• Check for referential integrity violations.

4. Data Consistency & Quality Check:
• Number check, date check, precision check, null check.
• Ensure consistent date format across all tables.

5. Null Validation:
• Check columns designated as NOT NULL do not receive NULL values.

6. Duplicate Validation:
• Check duplicate records across primary key or composite keys in target table.

7. Date Validation:
• Validate From_Date <= To_Date.
• Proper date format, no junk dates (0000-00-00) or missing dates.

8. Full Data Validation using MINUS Query:
• Perform Source MINUS Target and Target MINUS Source to ensure complete bidirectional data accuracy.""",
        "ETL Test Scenarios"
    )

    return qa_list

if __name__ == "__main__":
    qa_dataset = create_qa_dataset()
    with open("qa_data.json", "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated qa_data.json with {len(qa_dataset)} Q&A pairs.")
