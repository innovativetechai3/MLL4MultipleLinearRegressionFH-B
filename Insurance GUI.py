# Import required libraries
import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np
import warnings
from PIL import Image, ImageTk
from tkinter import messagebox

warnings.filterwarnings("ignore")

# *******************1 Load Saved Model and Scaler

# Load trained Multiple Linear Regression model
model = pickle.load(open("insurance_model.pkl", "rb"))

# Load StandardScaler used during training
scaler = pickle.load(open("scaler.pkl", "rb"))

# *******************2 Create Main Window

root = tk.Tk()

root.title("Insurance Charges Prediction System")
# Always appear in the center of the screen.
window_width = 700
window_height = 750

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width - window_width) // 2

y = (screen_height - window_height) // 2 - 35

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
# Prevent resizing
root.resizable(False, False)

# Background color
root.configure(bg="#F5F5F5")

# *******************3 Heading

header = tk.Frame(root, bg="#1565C0")
header.pack(fill="x")

# Load Logo
logo_img = Image.open("Logo.png")
logo_img = logo_img.resize((55,55))
logo = ImageTk.PhotoImage(logo_img)

logo_label = tk.Label(
    header,
    image=logo,
    bg="#1565C0"
)

logo_label.pack(side="left", padx=(15,10), pady=10)

heading = tk.Label(
    header,
    text="Insurance Charges Prediction System",
    bg="#1565C0",
    fg="white",
    font=("Helvetica",20,"bold")
)

heading.pack(side="left", pady=15)

# *******************4 Main Frame

frame = tk.Frame(
    root,
    bg="white",
    bd=3,
    relief="ridge",
    padx=35,
    pady=25
)

frame.pack(
    pady=18,
    padx=25
)

# *******************5 AGE

tk.Label(
    frame,
    text="Age",
    width=10,  #
    bg="white",
    font=("Calibri", 12)
).grid(row=0, column=0, padx=10, pady=10, sticky="w")

age_entry = tk.Entry(
    frame,
    width=15,#30,#28,
    font=("Calibri", 11)
)

age_entry.grid(
    row=0,
    column=1,
    padx=(15,5),
    pady=10,
    sticky="w"
)

# *******************6 GENDER

tk.Label(
    frame,
    text="Gender",
    width=10,
    bg="white",
    font=("Calibri", 12)
).grid(row=1, column=0, padx=10, pady=10, sticky="w")

gender_combo = ttk.Combobox(
    frame,
    width=15,
    state="readonly"
)

gender_combo["values"] = (
    "Male",
    "Female"
)

gender_combo.current(0)

gender_combo.grid(
    row=1, #0,
    column=1,
    padx=(15,5),
    pady=10,
    sticky="w"
)

# *******************7 BMI

tk.Label(
    frame,
    text="BMI",
    width=10,
    bg="white",
    font=("Calibri", 12)
).grid(row=2, column=0, padx=10, pady=10, sticky="w")

bmi_entry = tk.Entry(
    frame,
    width=15,
    font=("Calibri", 11)
)

# bmi_entry.grid(row=2, column=1)
bmi_entry.grid(
    row=2,
    column=1,
    padx=(15,5),
    pady=10,
    sticky="w"
)

# *******************8 CHILDREN

tk.Label(
    frame,
    text="Children",
    width=10,  #
    bg="white",
    font=("Calibri", 12)
).grid(row=3, column=0, padx=10, pady=10, sticky="w")

children_entry = tk.Entry(
    frame,
    width=15,#30, #28,
    font=("Calibri", 11)
)

children_entry.grid(
    row=3, #0,
    column=1,
    padx=(15,5),
    pady=10,
    sticky="w"
)

# *******************9 SMOKER

tk.Label(
    frame,
    text="Smoker",
    width=10,  #
    bg="white",
    font=("Calibri", 12)
).grid(row=4, column=0, padx=10, pady=10, sticky="w")

smoker_combo = ttk.Combobox(
    frame,
    width=15,
    state="readonly"
)

smoker_combo["values"] = (
    "Yes",
    "No"
)

smoker_combo.current(0)

smoker_combo.grid(
    row=4, #0,
    column=1,
    padx=(15,5),
    pady=10,
    sticky="w"
)

# *******************10 REGION

tk.Label(
    frame,
    text="Region",
    width=10,  #
    bg="white",
    font=("Calibri", 12)
).grid(row=5, column=0, padx=10, pady=10, sticky="w")

region_combo = ttk.Combobox(
    frame,
    width=15,
    state="readonly"
)

# All four regions from the original dataset
region_combo["values"] = (
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest"
)

region_combo.current(0)

region_combo.grid(
    row=5, #0,
    column=1,
    padx=(15,5),
    pady=10,
    sticky="w"
)

# *******************11 BMI Category (Display Only)

bmi_category_label = tk.Label(
    frame,
    text="BMI Category : Not Calculated",
    bg="white",
    fg="darkgreen",
    font=("Calibri", 11, "bold")
)

bmi_category_label.grid(
    row=6,
    column=0,
    columnspan=2,
    pady=15
)

# *******************12 Prediction Result

result_label = tk.Label(
    root,
    text="Predicted Insurance Charges",
    bg="#F5F5F5",
    fg="#1565C0",
    font=("Calibri", 16, "bold")
)

result_label.pack(
    pady=(4,4)
    # pady=(20,10)
)

# *******************13 Status Bar

status = tk.Label(
    root,
    text="Ready",
    anchor="w",
    relief="sunken",
    bg="#EEEEEE"
)

# *******************14 Prediction Function

def predict_charges():
    """
    Read user input, preprocess it exactly like the
    training pipeline, predict insurance charges,
    and display the result.
    """

    try:
        # Read User Input

        age = float(age_entry.get())
        bmi = float(bmi_entry.get())
        children = float(children_entry.get())

        gender = gender_combo.get()
        smoker = smoker_combo.get()
        region = region_combo.get()

        # Input Validation

        if age <= 0:
            messagebox.showerror(
                "Input Error",
                "Age must be greater than zero."
            )
            return

        if bmi <= 0:
            messagebox.showerror(
                "Input Error",
                "BMI must be greater than zero."
            )
            return

        if children < 0:
            messagebox.showerror(
                "Input Error",
                "Children cannot be negative."
            )
            return

        # Gender Encoding
        # Male   -> 1
        # Female -> 0

        if gender == "Male":
            is_gender = 1
        else:
            is_gender = 0

        # Smoker Encoding
        # Yes -> 1
        # No  -> 0
        # No  -> 0

        if smoker == "Yes":
            is_smoker = 1
        else:
            is_smoker = 0

        # Region Encoding
        # Model uses:
        # region_southeast
        # region_southwest
        # Northeast and Northwest become (0,0)

        region_southeast = 0
        region_southwest = 0

        if region == "Southeast":
            region_southeast = 1

        elif region == "Southwest":
            region_southwest = 1

        # BMI Category
        # Same logic used in training

        bmi_cat_HealthyWeight = 0
        bmi_cat_Obesity = 0

        if bmi < 18.5:

            category = "Underweight"

        elif bmi <= 24.9:

            category = "Healthy Weight"

            bmi_cat_HealthyWeight = 1

        elif bmi <= 29.9:

            category = "Overweight"

        else:

            category = "Obesity"

            bmi_cat_Obesity = 1

        bmi_category_label.config(
            text=f"BMI Category : {category}"
        )

        # Feature Scaling:
        # Scale only:
        # age
        # bmi
        # children

        scaled_values = scaler.transform(
            [[age, bmi, children]]
        )

        age_scaled = scaled_values[0][0]
        bmi_scaled = scaled_values[0][1]
        children_scaled = scaled_values[0][2]


        # Create Feature Vector
        # IMPORTANT
        # Feature order MUST match training.

        features = [[

            age_scaled,

            is_gender,

            bmi_scaled,

            children_scaled,

            is_smoker,

            region_southeast,

            bmi_cat_Obesity,

            region_southwest,

            bmi_cat_HealthyWeight

        ]]

        # Prediction

        prediction = model.predict(features)[0]

        # Display Prediction

        result_label.config(

            text=f"Predicted Insurance Charges\nRs. {prediction:,.2f}",

            fg="darkgreen"

        )

        status.config(
            text="Prediction Successful"
        )

    except ValueError:

        messagebox.showerror(

            "Input Error",

            "Please enter valid numeric values."

        )

    except Exception as e:

        messagebox.showerror(

            "Prediction Error",

            str(e)

        )

# *******************15 Reset Function

def reset_fields():

    age_entry.delete(0, tk.END)

    bmi_entry.delete(0, tk.END)

    children_entry.delete(0, tk.END)

    gender_combo.current(0)

    smoker_combo.current(0)

    region_combo.current(0)

    bmi_category_label.config(
        text="BMI Category : Not Calculated"
    )

    result_label.config(
        text="Predicted Insurance Charges",
        fg="#1565C0"
    )

    status.config(
        text="Fields Reset"
    )

# *******************16 Exit Function

def exit_app():

    answer = messagebox.askyesno(

        "Exit",

        "Do you really want to exit?"

    )

    if answer:

        root.destroy()

# *******************17 Predict Button

predict_btn = tk.Button(
    root,
    text="Predict Charges",
    command=predict_charges,
    width=22,
    height=1,
    bg="#2E7D32",
    fg="white",
    font=("Calibri", 13, "bold"),
    cursor="hand2"
)

predict_btn.pack(pady=(5, 10))

# *******************18 Reset Button

reset_btn = tk.Button(
    root,
    text="Reset",
    command=reset_fields,
    width=22,
    bg="#EF6C00",
    fg="white",
    font=("Calibri", 13, "bold"),
    cursor="hand2"
)

reset_btn.pack(pady=5)

# *******************19 Exit Button

exit_btn = tk.Button(
    root,
    text="Exit",
    command=exit_app,
    width=22,
    bg="#C62828",
    fg="white",
    font=("Calibri", 13, "bold"),
    cursor="hand2"
)

exit_btn.pack(pady=5)

# *******************20 Footer

footer = tk.Label(
    root,
    text="Developed by FH | @Multiple Linear Regression",
    bg="#1565C0",
    fg="white",
    font=("Calibri", 10)
)

footer.pack(
    side="bottom",
    fill="x",
    ipady = 3
)

# *******************21 Status Bar

status.pack(
    side="bottom",
    fill="x"
)

# *******************22 Start GUI

root.mainloop()