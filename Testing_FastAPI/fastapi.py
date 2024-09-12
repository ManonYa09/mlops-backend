from fastapi import FastAPI

app = FastAPI()

BOOKS = [
    {
        "title": "On the Road",
        "author": "Jack Kerouac",  # Corrected author name
        "year": 1951,
        "category": "fiction"
    },
    {
        "title": "Harry Potter and the Philosopher's Stone",
        "author": "J.K. Rowling",
        "year": 1997,
        "category": "fiction"
    }
]

@app.get("/books")
async def read_all_books():
    return BOOKS

@app.get("/books/{title}")
async def read_books_by_title(title: str):
    for book in BOOKS:
        if book["title"] == title:
            return book
    return {"data": "Not found"}

@app.get("/books/category")
async def read_category(category: str):
    books_to_return = []
    for book in BOOKS:
        if book.get('category').casefold() == category.casefold():
            books_to_return.append(book)
    return books_to_return