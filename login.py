import ipywidgets as widgets
from IPython.display import display
import requests
import os
from dotenv import load_dotenv

SUPABASE_URL = "https://udnmlcykctnahtnzmmmx.supabase.co"
# SUPABASE_KEY is now collected from the user

def login(email, password, supabase_key):
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=password"
    headers = {
        "apikey": supabase_key,
        "Content-Type": "application/json"
    }
    data = {"email": email, "password": password}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        json_response = response.json()
        jwt_token = json_response.get("access_token")
        user_id = json_response.get("user", {}).get("id")

        with open(".env", "a") as env_file:
            env_file.write(f"JWT={jwt_token}\n")
            env_file.write(f"UID={user_id}\n")
            env_file.write(f"SUPABASE_KEY={supabase_key}\n")
        return jwt_token, user_id
    else:
        print("Login failed:", response.json())
        return None, None

def login_form():
    # Create email, password and Supabase key fields
    email_input = widgets.Text(
        placeholder="Enter your email",
        description="Email:"
    )
    password_input = widgets.Password(
        placeholder="Enter your password",
        description="Password:"
    )
    supabase_key_input = widgets.Password(
        placeholder="Enter your Supabase key (from you .env file)",
        description="Supabase Key:",
        style={'description_width': 'initial'}
    )
    submit_button = widgets.Button(description="Submit")
    output = widgets.Output()

    # Function to handle submit button click
    def on_submit(b):
        with output:
            output.clear_output()
            email = email_input.value
            password = password_input.value
            supabase_key = supabase_key_input.value
            
            if not supabase_key:
                print("❌ Supabase key is required.")
                return
                
            jwt_token, user_id = login(email, password, supabase_key)
            if jwt_token:
                print("✅ Login successful!.")
            else:
                print("Please try again.")

    # Bind function to button click
    submit_button.on_click(on_submit)

    # Display widgets
    display(email_input, password_input, supabase_key_input, submit_button, output)
