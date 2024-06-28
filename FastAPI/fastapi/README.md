

# FastAPI Learning Management System: Shruti Mandaokar

This project is a Learning Management System (LMS) API built using FastAPI. It provides endpoints for managing users and retrieving user details.

## Features

- **Retrieve Users**: Get a list of all users.
- **Create User**: Add a new user to the system.
- **Retrieve User by ID**: Get user details by ID.

## Requirements

- Python 3.7+
- FastAPI
- Pydantic
- Uvicorn

## Installation

Install the required packages:
```bash
pip install fastapi uvicorn
```


## Start the FastAPI Server

To start the FastAPI server, run the following command:

```bash
uvicorn main:app --reload
```

# API Documentation

## Accessing the API Documentation
To view the interactive API documentation provided by Swagger UI, open your browser and go to:

## API Endpoints

### Retrieve Users
- **URL:** `/users`
- **Method:** `GET`
- **Response:** List of all users.

### Create User
- **URL:** `/users`
- **Method:** `POST`
- **Request Body:**
  
  ```json
 {
  "email": "user@example.com",
  "is_active": true,
  "bio": "Optional bio text"
}


### Retrieve User by ID
- **URL:** /users/{id}
- **Method:** GET
- Path Parameters:
id (int): The ID of the user you want to retrieve. Must be greater than 0.
- Query Parameters:
q (str): Optional query string with a maximum length of 5 characters.
- Response:
 Success message.
 User details or an error message if the user is not found.







