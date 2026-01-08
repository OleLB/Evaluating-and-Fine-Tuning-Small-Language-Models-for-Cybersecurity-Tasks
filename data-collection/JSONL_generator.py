import sqlite3
import json
import os

DB_PATH = "data-collection/cve_database.db"
OUTPUT_FOLDER = "CVE_records"

def export_to_jsonl(db_path=DB_PATH, output_dir=OUTPUT_FOLDER):
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        # 2. Connect to the database
        conn = sqlite3.connect(db_path)
        # Setting row_factory to sqlite3.Row allows us to access data by column name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 3. Fetch all records
        cursor.execute("SELECT * FROM cves")
        rows = cursor.fetchall()

        print(f"Starting export of {len(rows)} records...")

        for row in rows:
            # Convert the row object into a dictionary
            data = dict(row)
            
            # Use the cve_id as the filename
            filename = f"{data['cve_id']}.jsonl"
            file_path = os.path.join(output_dir, filename)

            # 4. Write to JSONL format
            with open(file_path, 'w', encoding='utf-8') as f:
                # json.dumps ensures the dictionary is a single string line
                f.write(json.dumps(data) + '\n')

        print(f"Export complete. Files saved in '{output_dir}/'")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    export_to_jsonl()