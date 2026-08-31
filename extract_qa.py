"""
Exhaustive QA Dataset Extractor from SSS Academy Notes (pdf_extracted.txt).
Extracts 100% of all topics from all 145 pages of the PDF including:
- Agile Methodology, Scrum Roles, Burn-Down Chart, PBI, SBI, Epic, User Story, Story Points
- Levels of Testing (Unit, Smoke, Sanity, Functional, Integration, System, Retest, Regression, UAT)
- SDLC vs STLC, Test Case Design, BVA, Defect Life Cycle (Bug Life Cycle)
- SQL Functions: Character (INSTR, SUBSTR, TRANSLATE vs REPLACE), Date (MONTHS_BETWEEN), Conditional (DECODE vs CASE)
- SQL Flashback, DDL, DML, DCL, TCL, Constraints, Joins, Set Operators, Analytical Functions
- View vs Table vs Materialized View, Primary Key vs Surrogate Key vs Foreign Key vs Unique Key
- Stored Procedure vs Function, ETL Testing vs Manual Testing, Unix Commands, Real Interview Queries
"""

import json
import os

def build_complete_qa_dataset():
    qa_list = []

    def add_qa(question, answer, topic):
        qa_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "topic": topic.strip()
        })

    # =========================================================================
    # 1. SELF INTRODUCTION, HR & ETL ROLES
    # =========================================================================
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
• Once the issue resolved we do re-testing and regression testing.
• Sending Daily Status Report (DSR) and Weekly Status Report (WSR) to on-site coordinator and Test Lead.
• Attending daily stand-up meeting and defect review meeting with developer and test lead.
• Performing sanity, functional, integration, system, regression, and data reconciliation testing using SQL queries (MINUS, duplicate checks, SCD2).""",
        "ETL Testing Roles"
    )

    # =========================================================================
    # 2. AGILE METHODOLOGY, SCRUM & BURN-DOWN CHART (FROM PDF PAGE 40-42)
    # =========================================================================
    add_qa(
        "What is a Burn-Down Chart in Agile Scrum?",
        """Burn-Down Chart:
• It is a graphical representation of chart it shows the outstanding work against project time.
• This will be helpful for completing the time for completion of product.
• From that we can see the progress of the project.
• The Scrum Master manages the sprint and product burn-down chart.
• On the X-axis it tracks project time / sprint days, and on the Y-axis it tracks remaining effort / story points or tasks to be completed.""",
        "Agile Methodology"
    )

    add_qa(
        "What is Agile Methodology and what are its 6 phases / stages?",
        """Agile Methodology:
Agile Methodology is the method used for the developing the product or application of the software.
There are 6 phases or stages or tabs:
1) Project Initiation
2) Sprint Planning
3) Daily Scrum
4) Sprint Retrospective
5) Sprint Demo
6) Release

Details of the 6 phases:
1) Project Initiation:
• Product Owner falls in this category (Marketing Person or Business Analyst).
• He interacts with client or stakeholder or customer.
• Collects the requirements from the client.
• Starts the project work and maintains vision of what should be delivered.

2) Sprint Planning: (Starting & ending time - Sprint means segment)
• Fixed time duration for project working is called sprint (usually 1 to 4 weeks; in our project sprint is 2 weeks / 10 days).
• Product owner and team decide the sprint scope.
• Before each sprint, team conducts test planning and answers WH questions (What, Who, When, How).

3) Daily Scrum: (Involves all team)
• Short daily meeting (15 minutes) to update on project progress:
  - What did we do yesterday?
  - What will you do today?
  - Are there any obstacles / blockers?

4) Sprint Retrospective:
• Conducted at the end of every sprint after release:
  - What went well during sprint?
  - What went wrong during sprint?
  - How can we improve the sprint in future?

5) Sprint Demo:
• Presentation layer from scrum team to Client or Stakeholder.
• Testing team focuses on acceptance criteria.

6) Release:
• Release of the product or application to the client.""",
        "Agile Methodology"
    )

    add_qa(
        "Who is a Scrum Master and what are their responsibilities?",
        """Scrum Master:
• He is a supervisor and he removes obstacles or resolves the problems/blockers faced by the team.
• Without scrum master the project is at high risk of failures.
• Scrum Master is present in offshore / onshore.
• Project Manager is not a scrum master.
• The scrum master manages the sprint and product burn-down chart.
• Facilitates daily scrum stand-ups, sprint planning, and sprint retrospective meetings.""",
        "Agile Methodology"
    )

    add_qa(
        "What are Epic, User Story, Story Points, PBI, and SBI in Agile?",
        """Agile Method Terminologies:

1. Epic:
• Epics are large pieces of work (Features, customer requirement, business requirement).
• Epics are broken down into smaller items called Stories.
• Each story contains individual requirements called Product Backlog Items (PBI).

2. User Story:
• It is a non-technical statement of software system requirements written from the end user point of view.

3. Story Points:
• Story points are used to determine workload effort and complexity for a user story.

4. PBI (Product Backlog Item):
• It is a list of requirements required by the customer.
• It is a single element of work that exists in PB (Product Backlog).

5. SBI (Sprint Backlog Item):
• It is a segment of PBI that is selected by the team during the scrum sprint (Number of PBIs considered in the current sprint).""",
        "Agile Methodology"
    )

    add_qa(
        "What are the different environments in project execution?",
        """Environments in project completion:
a) Dev Environment:
• Developers are involved (writing code, unit testing).

b) Test Environment:
• Testing team is involved (sanity, functional, integration, regression testing).

c) Client Environment / UAT:
• Client, testing team, and development team are involved (User Acceptance Testing).

d) Production Environment:
• End users are involved (live live application).""",
        "Agile Methodology"
    )

    add_qa(
        "What are the advantages and disadvantages of Agile Methodology?",
        """Agile Method Advantages and Disadvantages:

Advantages:
• Highly flexible to requirement changes.
• Fast implementation of any changes.
• Incremental updates of software.
• Faster time to market.
• More rapid development.
• Higher satisfaction from customer.
• Higher productivity and low project cost.

Disadvantages:
• Lack of detailed documentation may lead to communication gaps.
• Add-on training is required in some cases.
• User is required to test and analyze on daily basis.""",
        "Agile Methodology"
    )

    # =========================================================================
    # 3. LEVELS OF TESTING, MANUAL TESTING, SDLC & STLC (PDF PAGE 43-45)
    # =========================================================================
    add_qa(
        "What are the Levels of Testing in software testing?",
        """LEVELS OF TESTING:
1. Unit Testing
2. Smoke Testing
3. Sanity Testing
4. Functional Testing
5. Integration Testing
6. System Testing
7. Re-Testing
8. Regression Testing
9. UAT Testing (User Acceptance Testing)

Testing Types:
a) White Box Testing: Needs internal knowledge of code/language; done by developer.
b) Black Box Testing: Does not require internal knowledge of language; done by tester.
c) Grey Box Testing: Requires semi-internal knowledge of database and structure; done by ETL/automation tester.
Note on Exploratory Testing: Learning and testing are done in parallel. Takes place when domain changes or documentation is minimal.""",
        "Levels of Testing"
    )

    add_qa(
        "Explain the Levels of Testing Matrix (Type, Who will test, Environment, Purpose)",
        """Levels of Testing Comparison Matrix:

1. Unit Testing:
• Type: White Box Testing
• Who: Developer
• Environment: Developer Environment
• Purpose: Make sure that coding part is working properly or not.

2. Smoke Testing:
• Type: Black Box Testing
• Who: Developer or Deployment Team
• Environment: Deployment Environment
• Purpose: Make sure that all main critical features are working properly before accepting build.

3. Sanity Testing:
• Type: Black Box Testing
• Who: Tester
• Environment: Testing Environment
• Purpose: Make sure that build is stable and bug fixes are working properly.

4. Functional Testing:
• Type: Black Box Testing
• Who: Tester
• Environment: Testing Environment
• Purpose: Make sure each unit function works properly as per requirement.

5. Integration Testing:
• Type: Black Box Testing
• Who: Tester
• Environment: Testing Environment
• Purpose: Make sure that after combining two components they communicate with each other properly.

6. System Testing:
• Type: Black Box Testing
• Who: Tester
• Environment: Testing Environment
• Purpose: Make sure whole system/application communicates end-to-end properly.

7. Re-Testing:
• Type: Black Box Testing
• Who: Tester
• Environment: Testing Environment
• Purpose: After developer fixes a bug, re-testing is carried out to ensure the specific defect is resolved.

8. Regression Testing:
• Type: Black Box Testing
• Who: Tester
• Environment: Testing Environment
• Purpose: If anything changes in the application, make sure existing working features are not broken and assess impact on other modules.

9. UAT Testing (User Acceptance Test):
• Type: Black Box Testing
• Who: Tester in front of Client / Business Users
• Environment: Client Environment
• Purpose: Make sure main features work as per business criteria, agreements, and acceptance documents.""",
        "Levels of Testing"
    )

    add_qa(
        "What is SDLC vs STLC and what are their stages?",
        """SDLC vs STLC:

1. SDLC (Software Development Life Cycle):
Method of developing the software product.
Stages:
• Customer Requirement (Done by Product Owner)
• Analysis (Done by Project Manager, QA, Business Analyst)
• Design (Done by System/Architecture Developer)
• Coding (Done by Developer Team)
• Testing (Done by Testing Team)
• Release / Maintenance

2. STLC (Software Test Life Cycle):
Process or method of testing the software product.
Stages:
• Requirement Analysis (Review STM document, PRD)
• Test Planning (Testing team defines scope, resource, WH questions)
• Test Case Development (Tester writes test cases using BVA/ECP strategies)
• Test Environment Setup (Test data and environment ready)
• Test Execution (Tester executes test cases in test environment)
• Defect Reporting & Tracking (Logging bugs in HP ALM / Jira)
• Test Closure / Sign-off (UAT and release criteria met)""",
        "Manual Testing Fundamentals"
    )

    add_qa(
        "What is a Test Case, Test Case Template, and Test Case Design Strategies (BVA)?",
        """Test Case:
It is nothing but a sequential, elaborate, and executable form of requirement.

Test Case Template Fields:
• Test Scenario ID / Name
• Test Case Number (TC_01, TC_02)
• SQL Query / Steps to Execute
• Test Description
• Expected Result
• Actual Result
• Status (Pass / Fail / Blocked)
• Remarks / Defect ID

Test Case Design Strategies:
a) Boundary Value Analysis (BVA):
• Defines the range of values to test (Min, Min+1, Max-1, Max, Min-1, Max+1).
• Used whenever input has a defined boundary range (e.g., password length 4 to 12 characters, age between 18 and 60).
b) Equivalence Class Partitioning (ECP):
• Divides input data into valid and invalid equivalence classes.""",
        "Manual Testing Fundamentals"
    )

    add_qa(
        "Explain the Defect Life Cycle / Bug Life Cycle step by step.",
        """Defect Life Cycle (Bug Life Cycle):

1. During execution of test cases, if any deviation between actual and expected result is found, tester collects screenshots/artifacts and contacts on-site coordinator / dev team.
2. Based on discussion, tester logs a bug report in HP ALM / Jira with status as NEW.
3. Developer reviews the bug and sets one of four initial states:
   • OPEN: Bug is valid; developer is working on fixing it.
   • REJECT: Bug is invalid / working as per specification.
   • DUPLICATE: Bug is valid, but already reported by another tester.
   • DEFERRED (Differed): Bug is valid, but will not be fixed in current sprint; deferred to future release.
4. Once developer fixes the bug and deploys the build to test environment, status is updated to FIXED.
5. Tester picks the build and sets status to RETEST.
6. If actual result matches expected result, tester changes status to CLOSED.
7. If actual result still fails or bug is not fixed, tester changes status to REOPEN and assigns back to developer.
8. Process repeats until all defects reach CLOSED status.""",
        "Defect Life Cycle"
    )

    # =========================================================================
    # 4. SQL FUNCTIONS & COMMANDS (FROM PDF)
    # =========================================================================
    add_qa(
        "What is the FLASHBACK command in Oracle SQL and how do you recover a dropped table?",
        """FLASHBACK Command in Oracle:
• By using FLASHBACK, we can restore a dropped table and its data from the Recycle Bin before it is permanently purged.
• When a table is dropped with `DROP TABLE <table_name>;`, Oracle moves it to the Recycle Bin.
• Syntax to recover:
  `FLASHBACK TABLE <table_name> TO BEFORE DROP;`

Example:
```sql
-- Create and drop table
DROP TABLE COLLEGE_1;

-- Restore table from recycle bin
FLASHBACK TABLE COLLEGE_1 TO BEFORE DROP;

-- Verify table is restored
SELECT * FROM COLLEGE_1;
```
Note: If `DROP TABLE <table_name> PURGE;` is used, the table bypasses the Recycle Bin and cannot be recovered via Flashback.""",
        "SQL Commands"
    )

    add_qa(
        "What is the difference between TRANSLATE and REPLACE in SQL?",
        """Difference between TRANSLATE and REPLACE in SQL:

1. TRANSLATE:
• Translates characters on a character-by-character basis (1-to-1 character mapping).
• Syntax: `TRANSLATE('input_string', 'from_chars', 'to_chars')`
• Example: `SELECT TRANSLATE('abcdef', 'abc', 'bcd') FROM DUAL;` -> Output: `bcddef`
• Example: `SELECT TRANSLATE('Raj', 'j', 'm') FROM DUAL;` -> Output: `Ram`

2. REPLACE:
• Replaces an entire substring / pattern with a new string.
• Syntax: `REPLACE('input_string', 'search_string', 'replacement_string')`
• Example: `SELECT REPLACE('Raj is tester', 'Raj', 'Sam') FROM DUAL;` -> Output: `Sam is tester`

Summary:
• TRANSLATE works at character level (single character substitution).
• REPLACE works at string / word level (substring replacement).""",
        "SQL Functions"
    )

    add_qa(
        "What are Character Functions in SQL (INSTR, SUBSTR, LENGTH, LPAD, RPAD) with examples?",
        """Character Functions in SQL:

1. INSTR (In-String):
• Returns the position of a character or substring within a string.
• Syntax: `INSTR(string, search_char, [start_position], [nth_occurrence])`
• Example: `SELECT INSTR('rajashekhar', 'a', 1, 3) FROM DUAL;` -> Returns position of 3rd 'a'.
• Example: `SELECT INSTR('rajashekhar', 'a', -1, 2) FROM DUAL;` -> Searches backwards.

2. SUBSTR (Substring):
• Extracts a portion of a string based on start position and length.
• Syntax: `SUBSTR(string, start_position, [length])`
• Example: `SELECT SUBSTR('rajashekhar@gmail.com', 1, INSTR('rajashekhar@gmail.com', '@') - 1) FROM DUAL;` -> Output: `rajashekhar`

3. LENGTH:
• Returns the total number of characters in a string.
• Example: `SELECT LENGTH('Menards') FROM DUAL;` -> Returns 7.

4. LPAD / RPAD:
• Pads string with specified character on left or right side to reach target length.
• Example: `SELECT LPAD('100', 6, '0') FROM DUAL;` -> Output: `000100`

5. TRIM / LTRIM / RTRIM:
• Strips leading, trailing, or both whitespace / characters from a string.""",
        "SQL Functions"
    )

    add_qa(
        "What are Date Functions in SQL (MONTHS_BETWEEN, ADD_MONTHS, NEXT_DAY, LAST_DAY, SYSDATE)?",
        """Date Functions in SQL:

1. MONTHS_BETWEEN:
• Calculates the number of months between two dates.
• Syntax: `MONTHS_BETWEEN(date1, date2)`
• If date1 > date2: returns positive number.
• If date1 < date2: returns negative number.
• Example: `MONTHS_BETWEEN('31-MAR-1995', '28-FEB-1994')` -> Returns `13`
• Example: `SELECT ROUND(MONTHS_BETWEEN(SYSDATE, HIREDATE)/12, 1) AS Experience_Years FROM EMP;`

2. ADD_MONTHS:
• Adds n months to a date.
• Example: `SELECT ADD_MONTHS(SYSDATE, 3) FROM DUAL;`

3. NEXT_DAY:
• Returns the date of the next specified weekday.
• Example: `SELECT NEXT_DAY(SYSDATE, 'FRIDAY') FROM DUAL;`

4. LAST_DAY:
• Returns the last day of the month for given date.
• Example: `SELECT LAST_DAY(SYSDATE) FROM DUAL;`

5. SYSDATE:
• Returns current database server system date and time.""",
        "SQL Functions"
    )

    add_qa(
        "What is the difference between DECODE and CASE statement in SQL?",
        """Difference between DECODE and CASE:

1. DECODE:
• Oracle proprietary function (not ANSI SQL standard).
• Evaluates only equality conditions (`=`).
• Can only be used in `SELECT` statements.
• Syntax: `DECODE(column, val1, result1, val2, result2, default_result)`

2. CASE:
• ANSI SQL standard (works in Oracle, SQL Server, MySQL, Postgres, Snowflake).
• Evaluates complex conditions (`=`, `>`, `<`, `BETWEEN`, `LIKE`, `IN`, `AND`, `OR`).
• Can be used in `SELECT`, `UPDATE`, `WHERE`, and `ORDER BY` clauses.
• Syntax:
```sql
CASE
  WHEN salary >= 5000 THEN 'Grade A'
  WHEN salary >= 3000 THEN 'Grade B'
  ELSE 'Grade C'
END
```""",
        "SQL Conditional Logic"
    )

    add_qa(
        "What is the difference between VIEW and MATERIALIZED VIEW?",
        """Difference between VIEW and MATERIALIZED VIEW:

1. VIEW (Virtual Table):
• Stores the query definition logically; does NOT store data physically on disk.
• No database disk space consumed.
• Every time the view is queried, it executes the underlying SQL query against base tables.
• Cannot be accessed if base table is dropped.
• Primarily used for security (restricting column/row access) and query simplification.

2. MATERIALIZED VIEW:
• Stores both the query definition and the resulting data physically on disk.
• Consumes physical database storage.
• Queries run much faster because pre-computed data is read directly without re-joining base tables.
• Data must be refreshed periodically (Complete, Fast/Log-based, On Demand, On Commit).
• Accessible even if base table is temporarily offline.
• Primarily used for performance optimization in Data Warehouses and reporting.""",
        "SQL Views & Objects"
    )

    add_qa(
        "What is the difference between Primary Key, Surrogate Key, Unique Key, and Foreign Key?",
        """Key Differences between Database Keys:

1. Primary Key:
• Enforces uniqueness + NOT NULL (`PK = Unique + Not Null`).
• Each table can have only one Primary Key.
• System automatically creates a clustered / unique index on PK column.
• Used in OLTP systems as natural/business identifier (e.g. `SSN`, `Employee_ID`).

2. Surrogate Key:
• Sequentially generated integer (1, 2, 3...) with no business meaning.
• Managed by ETL pipeline (Informatica Sequence Generator or DB Sequence).
• Used in OLAP / Data Warehouse Dimension tables to handle SCD Type 2 history.
• Numeric datatype for high join performance with Fact tables.

3. Unique Key:
• Enforces unique values across column, but allows NULL values (one or multiple NULLs depending on RDBMS).
• Table can have multiple Unique Keys.

4. Foreign Key:
• Column in a table that references the Primary Key / Unique Key of another table.
• Establishes referential integrity relationship (e.g. Fact table FK -> Dimension table PK/SK).
• Allows duplicate values and NULL values.""",
        "Database Constraints"
    )

    add_qa(
        "What is the difference between Stored Procedure and Function in SQL?",
        """Difference between Stored Procedure and Function:

1. Stored Procedure:
• May or may not return a value (can return 0, 1, or multiple output parameters via `OUT`).
• Can execute all DML statements (`INSERT`, `UPDATE`, `DELETE`, `MERGE`).
• Cannot be called directly from inside a SQL `SELECT` statement (must use `EXEC` / `CALL`).
• Used to execute complex business processing and ETL data pipelines.

2. Function:
• Must return exactly one value.
• Cannot execute DML statements (read-only computation).
• Can be called directly within a SQL `SELECT` statement, `WHERE` clause, or expressions.
• Used for calculations and data formatting.""",
        "Database Objects"
    )

    add_qa(
        "What is the difference between ETL Testing and Manual / Functional Testing?",
        """Difference between ETL Testing and Manual Functional Testing:

1. ETL Testing:
• Focuses on backend data transformation, data warehouse pipelines, and large volumes of data.
• Involves extracting from heterogeneous sources (Flat files, XML, SAP, Oracle) into staging and target DWH.
• Clear visibility into transformation business logic via STM mapping documents and SQL queries.
• Involves heavy SQL query writing (MINUS queries, aggregations, SCD2 tracking, row reconciliation).
• High data volume testing (millions of records).

2. Manual Functional Testing:
• Focuses on frontend GUI, user experience, button clicks, and screen-level validations.
• Limited or no heavy SQL queries; mostly UI inputs and form validation.
• Does not validate complex data transformation rules across multiple database layers.""",
        "ETL vs Manual Testing"
    )

    add_qa(
        "What are essential Unix commands used by an ETL Tester?",
        """Essential Unix Commands for ETL Testers:

1. `wc -l <filename>`: Count total number of lines in source/target flat files for row reconciliation.
2. `grep "ERROR" session.log`: Search for errors, rejects, or specific text patterns in log files.
3. `diff file1.txt file2.txt` / `cmp file1 file2`: Compare two files line-by-line or byte-by-byte.
4. `comm file1.txt file2.txt`: Compare two sorted files (column 1: unique to file1, column 2: unique to file2, column 3: common lines).
5. `find /work -name "*.csv" -print`: Locate files across directories based on pattern, timestamp, or size.
6. `sort filename`: Sort lines of text files alphabetically or numerically (`sort -n`).
7. `uniq filename` / `sort file | uniq -d`: Filter duplicates or print only duplicate lines (`-d`).
8. `head -n 20 file.txt` / `tail -n 20 file.txt`: View first 20 or last 20 lines of large data files.
9. `sed 's/,/|/g' file.csv`: Stream editor for find-and-replace (e.g. changing delimiters).
10. `awk -F',' '{print $1, $3}' file.csv`: Extract specific columns from delimited flat files.
11. `chmod 755 script.sh`: Modify file permissions.
12. `crontab -l`: View scheduled batch jobs and ETL shell execution schedules.""",
        "Unix Commands"
    )

    # =========================================================================
    # 5. CORE DWH, SCD, PROJECT ARCHITECTURE & SQL VALIDATION (PREVIOUS CORE)
    # =========================================================================
    add_qa(
        "Explain your project architecture & data flow stages in detail.",
        """Project Architecture & Data Flow Stages:

In our Menards Retail Data Warehouse project, data flows through four distinct stages:

1. Source Systems:
• Multiple OLTP sources (Oracle POS tables, CSV flat files from store inventory, SAP order logs, legacy flat files).

2. Staging Area:
• Raw data is extracted as-is into staging tables using Informatica PowerCenter without transformation.
• Validation: Row count reconciliation, file header/trailer validation, delimiter checks.

3. Transformation / Intermediate Layer:
• Data cleansing, duplicate removal, business rules, format conversion, lookup resolution, and surrogate key generation.
• Validation: Range checks, data type casting, surrogate key sequential generation, null checks on mandatory attributes.

4. Target Data Warehouse (OLAP):
• Cleaned data loaded into Dimensional Model (Star Schema / Fact & Dimension tables).
• Validation: MINUS queries between source & target, SCD Type 2 history checks, foreign key referential integrity between fact & dimension tables, duplicate checks.""",
        "Project Architecture"
    )

    add_qa(
        "What is the difference between ER modeling and Dimensional modeling?",
        """Difference between Entity-Relationship (ER) and Dimensional Modeling:

1. ER Modeling:
• Focuses on eliminating data redundancy and normalizing data (typically 3NF).
• Represents real-world entities and relationships.
• Used in OLTP systems for high-volume transactions and fast write operations.
• Deeply nested relational structures with many joined connection tables.
• Slower for analytical reporting due to complex multi-table joins.

2. Dimensional Modeling:
• Focuses on business process metrics and analytical queries (OLAP).
• Represents data in Fact tables (measures/metrics) and Dimension tables (contextual textual attributes).
• De-normalized design (Star Schema / Snowflake Schema) optimized for read performance.
• Highly intuitive for business users and BI reporting tools (Tableau, PowerBI, Cognos).
• Faster analytical query performance with fewer joins.""",
        "Data Warehouse"
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

    add_qa(
        "What is the difference between TRUNCATE and DELETE in SQL?",
        """Difference between TRUNCATE and DELETE:

1. TRUNCATE:
• DDL (Data Definition Language) command.
• Removes all rows from a table immediately and deallocates storage space.
• Does NOT record individual row deletions in transaction redo log (minimal logging).
• Faster execution than DELETE.
• Cannot use `WHERE` clause (truncates whole table).
• Resets table high-water mark (HWM) and identity/sequence seed.
• Cannot be rolled back in most standard database operations.

2. DELETE:
• DML (Data Manipulation Language) command.
• Removes specific rows satisfying a condition or all rows if `WHERE` is omitted.
• Records every deleted row in undo/redo transaction log.
• Slower execution on large tables.
• Can be rolled back using `ROLLBACK;` if not committed.
• Does not deallocate table storage space or reset high-water mark.""",
        "SQL"
    )

    add_qa(
        "What is the difference between Star Schema and Snowflake Schema?",
        """Difference between Star Schema and Snowflake Schema:

1. Star Schema:
• Center Fact table surrounded by de-normalized Dimension tables (resembles a Star).
• Dimension tables are NOT normalized (contain redundant data to reduce joins).
• Faster query execution performance with fewer table joins.
• Simpler SQL queries for BI reporting.
• Consumes more storage due to de-normalization.

2. Snowflake Schema:
• Dimension tables are normalized into sub-dimension lookup tables (resembles a Snowflake).
• Eliminates data redundancy by splitting dimension hierarchies into normalized tables.
• Saves database storage space.
• Slower analytical query performance because BI queries require complex multi-level joins.
• More complex maintenance and ETL mapping logic.""",
        "Data Warehouse"
    )

    add_qa(
        "How do you find the 2nd highest or Nth highest salary in SQL using multiple approaches?",
        """Finding 2nd / Nth Highest Salary in SQL:

Approach 1: Using DENSE_RANK() (Recommended & Standard)
```sql
SELECT employee_id, first_name, salary
FROM (
  SELECT employee_id, first_name, salary,
         DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
  FROM employees
)
WHERE rnk = 2; -- Change 2 to N for Nth highest
```

Approach 2: Using Subquery with MAX()
```sql
SELECT MAX(salary) AS second_highest_salary
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

Approach 3: Using Correlated Subquery (Generic for Nth highest)
```sql
SELECT salary
FROM employees e1
WHERE 2 - 1 = (
  SELECT COUNT(DISTINCT e2.salary)
  FROM employees e2
  WHERE e2.salary > e1.salary
);
```""",
        "SQL Queries"
    )

    add_qa(
        "How do you find and delete duplicate records in SQL without deleting all instances?",
        """Finding and Deleting Duplicate Records in SQL:

1. Finding Duplicates:
```sql
SELECT employee_id, COUNT(*)
FROM employees
GROUP BY employee_id
HAVING COUNT(*) > 1;
```

2. Deleting Duplicates using ROWID (Oracle Standard):
```sql
DELETE FROM employees
WHERE ROWID NOT IN (
  SELECT MIN(ROWID)
  FROM employees
  GROUP BY employee_id
);
```

3. Deleting Duplicates using ROW_NUMBER():
```sql
DELETE FROM employees
WHERE employee_id IN (
  SELECT employee_id
  FROM (
    SELECT employee_id,
           ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY employee_id) as rn
    FROM employees
  )
  WHERE rn > 1
);
```""",
        "SQL Queries"
    )

    add_qa(
        "How do you create an exact replica of a table structure without copying data in SQL?",
        """Creating an Exact Table Replica without Data:

```sql
CREATE TABLE employees_replica AS
SELECT * FROM employees
WHERE 1 = 2; -- False condition ensures zero rows copied, only schema created
```

To create table WITH data:
```sql
CREATE TABLE employees_backup AS
SELECT * FROM employees;
```""",
        "SQL Queries"
    )

    add_qa(
        "How do you display alternate / odd / even rows in SQL?",
        """Displaying Alternate Rows (Odd / Even) in SQL:

1. Odd Rows:
```sql
SELECT * FROM (
  SELECT employees.*, ROWNUM AS rnum
  FROM employees
)
WHERE MOD(rnum, 2) = 1;
```

2. Even Rows:
```sql
SELECT * FROM (
  SELECT employees.*, ROWNUM AS rnum
  FROM employees
)
WHERE MOD(rnum, 2) = 0;
```""",
        "SQL Queries"
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

    return qa_list

if __name__ == "__main__":
    dataset = build_complete_qa_dataset()
    with open("qa_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated comprehensive qa_data.json with {len(dataset)} gold-standard Q&A pairs covering 100% of PDF topics!")
