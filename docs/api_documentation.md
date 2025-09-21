# API Documentation

Αυτό το έγγραφο περιγράφει το REST API της εφαρμογής.

## 🔗 Base URL

```
Production: https://api.yourdomain.com
Development: http://localhost:3001/api
```

## 🔐 Authentication

Το API χρησιμοποιεί JWT tokens για authentication.

### Λήψη Token

```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": 1,
      "email": "user@example.com",
      "name": "John Doe"
    }
  }
}
```

### Χρήση Token

Όλα τα protected endpoints χρειάζονται το Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 👤 Authentication Endpoints

### Register User

```http
POST /auth/register
Content-Type: application/json

{
  "name": "John Doe",
  "email": "user@example.com",
  "password": "password123",
  "confirmPassword": "password123"
}
```

**Response (201):**
```json
{
  "success": true,
  "message": "User registered successfully",
  "data": {
    "user": {
      "id": 1,
      "name": "John Doe",
      "email": "user@example.com",
      "createdAt": "2025-01-15T10:30:00Z"
    }
  }
}
```

### Login

```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

### Logout

```http
POST /auth/logout
Authorization: Bearer {token}
```

### Refresh Token

```http
POST /auth/refresh
Authorization: Bearer {refresh_token}
```

## 👥 User Endpoints

### Get Current User

```http
GET /users/me
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "John Doe",
    "email": "user@example.com",
    "avatar": "https://example.com/avatar.jpg",
    "createdAt": "2025-01-15T10:30:00Z",
    "updatedAt": "2025-01-20T14:20:00Z"
  }
}
```

### Update User Profile

```http
PUT /users/me
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "John Smith",
  "email": "newmail@example.com"
}
```

### Get All Users (Admin)

```http
GET /users?page=1&limit=10&search=john
Authorization: Bearer {admin_token}
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 10, max: 100)
- `search` (optional): Search term
- `sort` (optional): Sort field (name, email, createdAt)
- `order` (optional): Sort order (asc, desc)

## 📝 Data Endpoints

### Get Items

```http
GET /items?page=1&limit=10&category=tech
Authorization: Bearer {token}
```

**Query Parameters:**
- `page` (optional): Page number
- `limit` (optional): Items per page
- `category` (optional): Filter by category
- `status` (optional): Filter by status (active, inactive)
- `search` (optional): Search in title and description

**Response (200):**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": 1,
        "title": "Sample Item",
        "description": "This is a sample item",
        "category": "tech",
        "status": "active",
        "userId": 1,
        "createdAt": "2025-01-15T10:30:00Z",
        "updatedAt": "2025-01-20T14:20:00Z"
      }
    ],
    "pagination": {
      "currentPage": 1,
      "totalPages": 5,
      "totalItems": 50,
      "hasNext": true,
      "hasPrev": false
    }
  }
}
```

### Get Single Item

```http
GET /items/{id}
Authorization: Bearer {token}
```

### Create Item

```http
POST /items
Authorization: Bearer {token}
Content-Type: application/json

{
  "title": "New Item",
  "description": "Item description",
  "category": "tech",
  "tags": ["javascript", "react"],
  "metadata": {
    "priority": "high"
  }
}
```

### Update Item

```http
PUT /items/{id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "title": "Updated Item",
  "description": "Updated description",
  "status": "inactive"
}
```

### Delete Item

```http
DELETE /items/{id}
Authorization: Bearer {token}
```

## 📁 File Upload Endpoints

### Upload File

```http
POST /upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: [binary data]
folder: "images" (optional)
```

**Response (200):**
```json
{
  "success": true,
  "data": {
    "url": "https://storage.example.com/files/abc123.jpg",
    "filename": "abc123.jpg",
    "originalName": "photo.jpg",
    "size": 1024576,
    "mimeType": "image/jpeg"
  }
}
```

### Delete File

```http
DELETE /upload/{filename}
Authorization: Bearer {token}
```

## 📊 Analytics Endpoints

### Get Dashboard Stats

```http
GET /analytics/dashboard
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "success": true,
  "data": {
    "totalUsers": 150,
    "totalItems": 1200,
    "activeItems": 800,
    "recentActivity": [
      {
        "action": "item_created",
        "userId": 1,
        "timestamp": "2025-01-20T14:20:00Z"
      }
    ]
  }
}
```

## 🔍 Search Endpoints

### Global Search

```http
GET /search?q=javascript&type=items&page=1&limit=10
Authorization: Bearer {token}
```

**Query Parameters:**
- `q` (required): Search query
- `type` (optional): Search type (items, users, all)
- `page` (optional): Page number
- `limit` (optional): Results per page

## ⚠️ Error Responses

Το API επιστρέφει συνεπή error responses:

### Validation Error (400)

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      },
      {
        "field": "password",
        "message": "Password must be at least 8 characters"
      }
    ]
  }
}
```

### Authentication Error (401)

```json
{
  "success": false,
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token"
  }
}
```

### Forbidden (403)

```json
{
  "success": false,
  "error": {
    "code": "FORBIDDEN",
    "message": "Insufficient permissions"
  }
}
```

### Not Found (404)

```json
{
  "success": false,
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found"
  }
}
```

### Server Error (500)

```json
{
  "success": false,
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Internal server error"
  }
}
```

## 📏 Rate Limiting

Το API εφαρμόζει rate limiting:

- **Anonymous requests**: 100 requests/hour
- **Authenticated users**: 1000 requests/hour
- **Premium users**: 5000 requests/hour

**Headers στο response:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642694400
```

## 🔧 API Versioning

Το API υποστηρίζει versioning μέσω URL:

```
/v1/users      # Version 1
/v2/users      # Version 2 (νέα features)
```

Η τρέχουσα έκδοση είναι v1. Η v2 είναι σε beta.

## 📚 SDK & Client Libraries

### JavaScript/TypeScript

```bash
npm install your-api-client
```

```javascript
import { ApiClient } from 'your-api-client';

const client = new ApiClient({
  baseURL: 'https://api.yourdomain.com',
  token: 'your-jwt-token'
});

// Usage
const users = await client.users.getAll();
const item = await client.items.create({
  title: 'New Item',
  description: 'Description'
});
```

### Python

```bash
pip install your-api-python
```

```python
from your_api import ApiClient

client = ApiClient(
    base_url='https://api.yourdomain.com',
    token='your-jwt-token'
)

# Usage
users = client.users.get_all()
item = client.items.create({
    'title': 'New Item',
    'description': 'Description'
})
```

## 🧪 Testing

### Postman Collection

Κατέβασε την Postman collection: [API Collection](./postman_collection.json)

### Example Requests

```bash
# cURL examples

# Login
curl -X POST https://api.yourdomain.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123"}'

# Get items
curl -X GET https://api.yourdomain.com/items \
  -H "Authorization: Bearer your-token"

# Create item
curl -X POST https://api.yourdomain.com/items \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"title":"New Item","description":"Description"}'
```

## 📞 Support

- **API Documentation**: [https://docs.yourdomain.com](https://docs.yourdomain.com)
- **Status Page**: [https://status.yourdomain.com](https://status.yourdomain.com)
- **Support Email**: api-support@yourdomain.com
- **Discord**: [Community Server](https://discord.gg/your-server)

## 📋 Changelog

### v1.2.0 (2025-01-20)
- Προσθήκη search endpoints
- Βελτιώσεις στο rate limiting
- Νέα analytics endpoints

### v1.1.0 (2025-01-15)
- Προσθήκη file upload
- Authentication improvements
- Bug fixes

### v1.0.0 (2025-01-10)
- Initial release
- Core CRUD operations
- JWT authentication