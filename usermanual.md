# User Manual

### Project Overview

The AI Medical Voice Assistant is a tool for healthcare professionals, providing a conversational interface to access medical information and perform calculations. This manual will guide you through the setup and usage of the application.

### Setup and Installation

Follow these steps to set up the project on your local machine:

1.  **Clone the Repository:**
    Use `git clone` to download the project source code.
    ```bash
    git clone [https://github.com/AndreIglesias/AI-Medical-Voice-Assistant.git](https://github.com/AndreIglesias/AI-Medical-Voice-Assistant.git)
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd AI-Medical-Voice-Assistant
    ```

3.  **Set up the Virtual Environment:**
    A virtual environment (`venv`) is created to isolate the project's dependencies from your system's Python packages.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    * **macOS/Linux:** `source venv/bin/activate`
    * **Windows:** `venv\Scripts\activate`

4.  **Install Dependencies:**
    Install all required packages from the `requirements.txt` file. The `-e .` flag installs the package in editable mode.
    ```bash
    pip install -e .
    ```

5.  **Create the `.env` File:**
    This file stores your sensitive API keys. In the main project directory, create a new file named `.env` and paste the following content:
    ```
    speech = fw_3ZfrrxLEQq441P9Jqw6kPLd1
    refine = fw_3ZURm9vXSrWS4nnPMo9K6CPG
    translation = fw_3ZPLkUDqtJnyXnM7Rn6KjUHg
    extraction = fw_3ZSDxLZKLtqAb1ctkWpffvaE
    ```

### Running and Testing the Application

1.  **Run the Flask Application:**
    Start the server by running the `flask_app.py` file from your terminal while inside the activated virtual environment.
    ```bash
    python flask_app.py
    ```
    The server will start, and you will see an output indicating the local address where it is running (e.g., `http://127.0.0.1:5000`).

2.  **Testing the `analyze` Endpoint:**
    As noted, the `analyze` endpoint is designed to stream output and is best tested using a command-line tool like `curl` rather than a client like Postman, which may not handle streaming responses correctly.

    Use the following `curl` command as a template to test the endpoint. Remember to replace `[YOUR_ENDPOINT_URL]` with the actual URL provided by your running Flask server and `[YOUR_DATA]` with the JSON data you want to send.

    ```bash
    curl -N -X POST [YOUR_ENDPOINT_URL]/analyze -H "Content-Type: application/json" -d '[YOUR_DATA]'
    ```
    The `-N` flag is important as it disables the buffering of the output, allowing you to see the streamed results one by one in your terminal.
