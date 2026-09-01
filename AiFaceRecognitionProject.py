import os
import cv2
import customtkinter as ctk
from tkinter import messagebox, simpledialog
import mysql.connector
import face_recognition
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
from openpyxl.utils import get_column_letter
# theme and colour setting
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ----GLOBAL VARIABLE FOR DYNAMIC DATABASE CONFIG ----
MYSQL_PASSWORD = ""

# ----STEP 1: DATABASE & TABLES AUTO-INITIALIZATION ----
def auto_setup_database():
    global MYSQL_PASSWORD
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS school_db")
        conn.commit()
        cursor.close()
        conn.close()

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=MYSQL_PASSWORD,
            database="school_db"
        )
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_name VARCHAR(100) NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT,
                student_name VARCHAR(100),
                attendance_date DATE,
                attendance_time TIME,
                status VARCHAR(10) DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as err:
        messagebox.showerror("Connection Failed", f"MySQL Connection Error!\nKripya password dobara check karein.\n\nDetails: {err}")
        return False

def get_db_connection():
    global MYSQL_PASSWORD
    return mysql.connector.connect(
        host="localhost", user="root", password=MYSQL_PASSWORD, database="school_db"
    )

# ----STEP 2: CAMERA CAPTURE & REGISTRATION ----
def register_student():
    student_name = simpledialog.askstring("Input Needed", "Enter Student Full Name:")
    if not student_name or student_name.strip() == "":
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_name, image_path FROM students")
    rows = cursor.fetchall()
    
    known_face_encodings = []
    known_face_names = []
    for row in rows:
        name, img_path = row
        if os.path.exists(img_path):
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])  # Use 1D array profile
                known_face_names.append(name)

    folder_name = "student_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    video_capture = cv2.VideoCapture(0)
    messagebox.showinfo("Instructions", "Camera chalu hoga. Snapshot ke liye keyboard par 's' dabayein.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break
        
        cv2.imshow("Registration - Press 's' to Save / 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            new_encodings = face_recognition.face_encodings(rgb_frame)
            
            if len(new_encodings) > 0:
                matches = face_recognition.compare_faces(known_face_encodings, new_encodings[0], tolerance=0.5)
                
                if True in matches:
                    match_index = matches.index(True)
                    existing_student = known_face_names[match_index]
                    messagebox.showerror("Duplicate Image Detected", f"रजिस्ट्रेशन कैंसिल! यह चेहरा पहले से ही '{existing_student}' के नाम से रजिस्टर्ड है।")
                    break
            
            clean_name = student_name.lower().replace(' ', '_')
            image_path = f"{folder_name}/{clean_name}.jpg"
            
            try:
                cursor.execute("INSERT INTO students (student_name, image_path) VALUES (%s, %s)", (student_name, image_path))
                conn.commit()
                cv2.imwrite(image_path, frame)
                messagebox.showinfo("Success", f"{student_name} successfully register ho gaya!")
                if callback_func is not None:
                    callback_func()
            except mysql.connector.Error as err:
                if err.errno == 1062:
                    messagebox.showerror("Error", f"यह नाम '{student_name}' पहले से डेटाबेस में मौजूद है!")
                else:
                    messagebox.showerror("SQL Error", str(err))
            break
            
        elif key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()
# ----STEP 3: REAL-TIME AI FACE RECOGNITION ----
def start_attendance_system():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, student_name, image_path FROM students")
    rows = cursor.fetchall()
    
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    for row in rows:
        s_id, name, img_path = row
        if os.path.exists(img_path):
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                known_face_ids.append(s_id)

    if not known_face_encodings:
        messagebox.showwarning("Warning", "Database me koi records nahi hain!\nPehle bache ko register karein.")
        cursor.close()
        conn.close()
        return

    video_capture = cv2.VideoCapture(0)
    already_marked_today = set()

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            student_id = None

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    student_id = known_face_ids[best_match_index]
            
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            box_color = (0, 255, 0)
            if name != "Unknown":
                display_text = f"{name} (ID: {student_id})"
            else:
                    display_text = "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame,display_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if name != "Unknown" and name not in already_marked_today:
                now = datetime.now()
                current_date = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M:%S")

                cursor.execute("SELECT * FROM attendance WHERE student_id = %s AND attendance_date = %s", (student_id, current_date))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO attendance (student_id, student_name, attendance_date, attendance_time) VALUES (%s, %s, %s, %s)", (student_id, name, current_date, current_time))
                    conn.commit()
                    print(f"[Attendance Log]: Present marked for {name}")
                already_marked_today.add(name)

        cv2.imshow("AI Face Attendance System - Press 'q' to Exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()

# ---- 📈 STEP 4: CLEAN DATA PANDAS EXCEL REPORT GENERATOR ----
def export_to_excel():
    conn = get_db_connection()
    query = """
        SELECT a.student_id AS 'Student ID', 
               a.student_name AS 'Student Name', 
               a.attendance_date AS 'Date', 
               a.attendance_time AS 'Time', 
               a.status AS 'Status'
        FROM attendance a
        INNER JOIN (
            SELECT student_id, attendance_date, MIN(attendance_time) as min_time
            FROM attendance
            GROUP BY student_id, attendance_date
        ) b 
        ON a.student_id = b.student_id 
        AND a.attendance_date = b.attendance_date 
        AND a.attendance_time = b.min_time
        ORDER BY a.attendance_date DESC
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            messagebox.showwarning("No Data", "Database me koi log record nahi mila!")
            return

        if 'Time' in df.columns:
            df['Time'] = df['Time'].astype(str).str.split().str[-1]

        today_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        file_name = f"Attendance_Report_{today_str}_{time_str}.xlsx"        
        
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']
            
            for col_idx, col_cells in enumerate(worksheet.columns, start=1):
                max_len = max(len(str(cell.value or '')) for cell in col_cells)
                col_letter = get_column_letter(col_idx)
                worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

        messagebox.showinfo("Export Successful", f"Excel sheet export successfully completed!\nFile: {file_name}")
    except Exception as e:
        messagebox.showerror("Export Error", str(e))

# ---- 5: DELETE STUDENT REGISTRATION & IMAGE ----
def delete_student():
    student_name = simpledialog.askstring("Delete Request", "Enter Student Full Name to Delete:")
    if not student_name or student_name.strip() == "":
        return

    # Admin confirmation
    confirm = messagebox.askyesno("Confirm Delete", f"Do yo want to delete {student_name} data?")
    if not confirm:
        return

    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)
    
    try:
        # check the already registered students image path and Id
        cursor.execute("SELECT id, image_path FROM students WHERE student_name = %s", (student_name,))
        row = cursor.fetchone()
        
        if row:
            student_id, image_path = row
            
            # Delete the attendance record
            cursor.execute("DELETE FROM attendance WHERE student_id = %s", (student_id,))
            
            # delete the profile
            cursor.execute("DELETE FROM students WHERE id = %s", (student_id,))
            
            conn.commit()
            
            # delete the Image
            if os.path.exists(image_path):
                os.remove(image_path)
                
            messagebox.showinfo("Deleted", f"{student_name} has been deleted")
        else:
            messagebox.showwarning("Not Found", f"'{student_name}' not found")
            
    except Exception as e:
        messagebox.showerror("Database Error", f"Error in deletion: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        
# ---- STEP 5: SHOW RECORDS OF REGISTERED STUDENTS ----
def registered_students():
    messagebox.showinfo("Exporting Text Data", "Click to confirm")
    conn = get_db_connection()
    query = "SELECT * FROM students"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_path = temp_file.name

    df.to_csv(temp_path, index=False, sep="|")

    if os.name == "nt":
        os.startfile(temp_path)
    else:
        import subprocess
        subprocess.call(["open" if os.name == "mac" else "xdg-open", temp_path])


# ================= 💻 NEW CUSTOMTKINTER UI LAYOUT =================

# ---- 📊 REGISTER DASHBOARD (MAIN WINDOW) ----

class AttendanceDashboard(ctk.CTk):
    def total_students(self):
        #self.messagebox.showinfo("Exporting Text Data", "Click to confirm")
        conn = get_db_connection()
        query = "SELECT * FROM students"
        cursor=conn.cursor()
        cursor.execute(query)
        all_students = cursor.fetchall()
        
        self.total_Student.configure(text=str( len(all_students))) 
    def refresh_tracked_table(self):
        # Clear dynamic frame
        for widget in self.table_frame.winfo_children():
            widget.destroy()
            self.table_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        # CHANGED: Table now structures matching historical layout criteria
        headers = ["Student ID", "Full Name", "Time Tracked", "Status"]
        for col_idx, text in enumerate(headers):
            lbl = ctk.CTkLabel(self.table_frame, text=text, font=ctk.CTkFont(weight="bold"), text_color="gray")
            lbl.grid(row=0, column=col_idx, padx=15, pady=10, sticky="w")
        try:
            conn = get_db_connection()
            cursor = conn.cursor(buffered=True)
            current_date = datetime.now().strftime("%Y-%m-%d")
            # CHANGED: Queries attendance logging instances instead of student static tables
            cursor.execute("""SELECT student_id, student_name, attendance_time,
status FROM attendance WHERE attendance_date = %s ORDER BY attendance_id DESC""", (current_date,))
            tracked_rows = cursor.fetchall()
            # Update counter panel live numbers mapping today's inputs
            self.total_student_lbl.configure(text=str(len(tracked_rows)))
            for row_idx, row_data in enumerate(tracked_rows, start=1):
                for col_idx, text in enumerate(row_data):
                    # Style modifications parsing string formatting parameters
                    if col_idx == 3:
                        # Status Column (Present text color set to bright green)
                        lbl = ctk.CTkLabel(self.table_frame, text=str(text), text_color="#2ecc71", font=ctk.CTkFont(weight="bold"))
                        lbl.grid(row=row_idx, column=col_idx, padx=15, pady=6, sticky="w")

                    else:
                        lbl = ctk.CTkLabel(self.table_frame, text=str(text), text_color="#ecf0f1" if col_idx != 0 else "#1abc9c")
                        lbl.grid(row=row_idx, column=col_idx, padx=15, pady=6, sticky="w")
                        cursor.close()
                        conn.close()
        except Exception as e:
            error_lbl = ctk.CTkLabel(self.table_frame, text=f"Failed to fetch track rows: {str(e)}", text_color="#e74c3c")
            error_lbl.grid(row=1, column=0, columnspan=3, padx=15, pady=10)
        self.after(3000, self.refresh_tracked_table)

    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System - Admin Dashboard")
        self.geometry("1100x650")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # ---- SIDEBAR NAVIGATION ----
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#1e1e24")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🤖 ADMIN PANEL", font=ctk.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=20)
        # buttons connected with the database
        self.btn_dash = ctk.CTkButton(self.sidebar, text="Dashboard (Home)", fg_color="#2a2a35", anchor="w")
        self.btn_dash.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.btn_reg_student = ctk.CTkButton(self.sidebar, text="1. New Registration", fg_color="transparent", anchor="w", command=register_student)
        self.btn_reg_student.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.btn_scan = ctk.CTkButton(self.sidebar, text="2. Start Live Tracker", fg_color="transparent", anchor="w", command=start_attendance_system)
        self.btn_scan.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.btn_rec = ctk.CTkButton(self.sidebar, text="3. Show Registered", fg_color="transparent", anchor="w", command=registered_students)
        self.btn_rec.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.btn_rep = ctk.CTkButton(self.sidebar, text="4. Generate Excel", fg_color="transparent", anchor="w", command=export_to_excel)
        self.btn_rep.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        self.btn_del_student = ctk.CTkButton(self.sidebar, text="5. Delete Registration", fg_color="transparent", hover_color="#7b241c", anchor="w", command=delete_student)
        self.btn_del_student.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        self.btn_logout = ctk.CTkButton(self.sidebar, text="🚪 Logout Screen", fg_color="transparent", text_color="#ff4d4d", anchor="w", command=self.logout)
        self.btn_logout.grid(row=7, column=0, padx=20, pady=20, sticky="ew")
        # ---- MAIN CONTENT AREA ----
        self.main_frame = ctk.CTkScrollableFrame(self, fg_color="#121214", corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure((0, 1, 2,3), weight=1)
        self.welcome_lbl = ctk.CTkLabel(self.main_frame, text="AI & Robotics Lab Portal Dashboard", font=ctk.CTkFont(size=24, weight="bold"))
        self.welcome_lbl.grid(row=0, column=0, columnspan=3, sticky="w", pady=10)
        # analytical Data matrix code
        self.card_total = ctk.CTkFrame(self.main_frame, fg_color="#1e1e24", height=100)
        self.card_total.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        #ctk.CTkLabel(self.card_present, text="Total", text_color="gray").pack(pady=5)
        ctk.CTkLabel(self.card_total, text="Total", font=ctk.CTkFont(size=20, weight="bold"), text_color="#2ecc71").pack()
        self.total_Student=ctk.CTkLabel(self.card_total,text="loading...",font=ctk.CTkFont(size=20,weight="bold"),text_color="#2ecc71")
        self.total_Student.pack()
        self.card_present = ctk.CTkFrame(self.main_frame, fg_color="#1e1e24", height=100)
        self.card_present.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        #ctk.CTkLabel(self.card_present, text="Total Active Student Logs", text_color="gray").pack(pady=5)
        ctk.CTkLabel(self.card_present, text="Present", font=ctk.CTkFont(size=20, weight="bold"), text_color="#2ecc71").pack()

        self.total_student_lbl = ctk.CTkLabel(self.card_present, text="Loading...", font=ctk.CTkFont(size=20, weight="bold"), text_color="#2ecc71")
        self.total_student_lbl.pack()

        self.card_late = ctk.CTkFrame(self.main_frame, fg_color="#1e1e24", height=100)
        self.card_late.grid(row=1, column=2, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self.card_late, text="System Core State", text_color="gray").pack(pady=5)
        ctk.CTkLabel(self.card_late, text="Verified", font=ctk.CTkFont(size=20, weight="bold"), text_color="#f1c40f").pack()
        self.card_absent = ctk.CTkFrame(self.main_frame, fg_color="#1e1e24", height=100)
        self.card_absent.grid(row=1, column=3, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self.card_absent, text="Database Status", text_color="gray").pack(pady=5)
        ctk.CTkLabel(self.card_absent, text="Connected", font=ctk.CTkFont(size=20, weight="bold"), text_color="#3498db").pack()
        # Activity log here
        self.log_title = ctk.CTkLabel(self.main_frame, text="Recent Live Activity Panel", font=ctk.CTkFont(size=18, weight="bold"))
        self.log_title.grid(row=2, column=0, columnspan=3, sticky="w", pady=10,padx=10)
        self.table_frame = ctk.CTkFrame(self.main_frame, fg_color="#1e1e24")
        self.table_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        self.table_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        headers = ["Status", "Server Host", "User Node", "Environment"]
        for col_idx, text in enumerate(headers):
            lbl = ctk.CTkLabel(self.table_frame, text=text, font=ctk.CTkFont(weight="bold"), text_color="gray")
            lbl.grid(row=0, column=col_idx, padx=10, pady=10, sticky="w")
        #to track the registered students
        
        lbl_status = ctk.CTkLabel(self.table_frame, text="OPERATIONAL", text_color="#2ecc71", font=ctk.CTkFont(weight="bold"))
        lbl_status.grid(row=1, column=0, padx=10, pady=8, sticky="w")
        lbl_host = ctk.CTkLabel(self.table_frame, text="localhost (127.0.0.1)")
        lbl_host.grid(row=1, column=1, padx=10, pady=8, sticky="w")
        lbl_user = ctk.CTkLabel(self.table_frame, text="root")
        lbl_user.grid(row=1, column=2, padx=10, pady=8, sticky="w")
        lbl_env = ctk.CTkLabel(self.table_frame, text="CustomTkinter Dark")
        lbl_env.grid(row=1, column=3, padx=10, pady=8, sticky="w")
         
        # ---- DYNAMIC HOOK: LIVE TRACKED PRESENT STUDENTS GRID PANEL ----
        self.log_title = ctk.CTkLabel(self.main_frame, text="📋 Tracked Present Students (Today's Logs)", font=ctk.CTkFont(size=18, weight="bold"))
        self.log_title.grid(row=4, column=0, columnspan=3, sticky="w", pady=(20, 10))
        # Dynamic Grid Container Frame
        self.table_frame = ctk.CTkFrame(self.main_frame, fg_color="#1e1e24")
        self.table_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        # Fire active data rendering loop
        self.refresh_tracked_table()
        self.total_students()
        # footer
        self.footer = ctk.CTkLabel(self.main_frame, text="System Core Verified - Operations Active", font=ctk.CTkFont(size=12, slant="italic"), text_color="#95a5a6")
        self.footer.grid(row=6, column=0, columnspan=3, pady=25)
    def logout(self):
        self.destroy()
        login_app = LoginWindow()
        login_app.mainloop()
        #----  DATABASE AUTHENTICATION GATEWAY (LOGIN WINDOW) ----
class LoginWindow(ctk.CTk):
    def attempt_login(self):
        global MYSQL_PASSWORD
        MYSQL_PASSWORD = self.password_entry.get()
        if auto_setup_database():
            self.destroy()
            dashboard_app = AttendanceDashboard()
            dashboard_app.mainloop()
    def __init__(self):
        super().__init__()
        self.title("MySQL Server Authentication")
        self.geometry("450x320")
        self.resizable(False, False)
    
    
        self.card = ctk.CTkFrame(self, width=390, height=280, corner_radius=12, border_width=2, border_color="#1abc9c")
        self.card.place(relx=0.5, rely=0.5, anchor="center")
        self.title_lbl = ctk.CTkLabel(self.card, text="DATABASE GATEWAY", font=ctk.CTkFont(size=20, weight="bold"), text_color="#1abc9c")
        self.title_lbl.place(relx=0.5, rely=0.15, anchor="center")
        self.user_lbl = ctk.CTkLabel(self.card, text="User: root (Local Admin)", text_color="gray")
        self.user_lbl.place(relx=0.5, rely=0.3, anchor="center")
        self.pass_lbl = ctk.CTkLabel(self.card, text="Enter MySQL Root Password:", font=ctk.CTkFont(weight="bold"))
        self.pass_lbl.place(relx=0.5, rely=0.45, anchor="center")
        # Password input field
        self.password_entry = ctk.CTkEntry(self.card, show="*", width=260, height=35, justify='center')
        self.password_entry.place(relx=0.5, rely=0.6, anchor="center")
        #focus on the password field
        self.password_entry.focus()
        self.after(40, lambda: self.password_entry.focus_force())

        # Button Connection
        self.btn_connect = ctk.CTkButton(self.card, text="CONNECT SERVER", fg_color="#1abc9c", hover_color="#16a085", width=200, height=38, font=ctk.CTkFont(weight="bold"), command=self.attempt_login)
        self.btn_connect.place(relx=0.5, rely=0.8, anchor="center")
        
if __name__ == "__main__":
    app = LoginWindow()
    app.mainloop()
