
def get_weather(city: str) -> str:
    """Get sample weather for a city."""
    weather = {
        "los angeles": "Sunny, 24°C",
        "new york": "Cloudy, 18°C",
        "baltimore": "Rainy, 20°C"
    }
    return weather.get(city.lower(), "City not found")


def search_books(query: str) -> str:
    """Search books by title or author."""
    books = {
        "1984": "George Orwell",
        "Values": "Mark Carney",
        "Atomic Habits": "James Clear"
    }
    return "\n".join(
        f"{title} by {author}"
        for title, author in books.items()
        if query.lower() in title.lower() or query.lower() in author.lower()
    ) or "No books found"


TOOLS = {
    "get_weather": get_weather,
    "search_books": search_books
}
