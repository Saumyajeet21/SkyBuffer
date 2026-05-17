# backend/airports_db.py
import csv, io, urllib.request, json
from pathlib import Path

CACHE = Path(__file__).parent.parent / "data" / "airports_cache.json"

INDIAN_AIRPORTS = {
    "DEL":{"name":"Indira Gandhi International","city":"New Delhi","country":"India","icao":"VIDP"},
    "BOM":{"name":"Chhatrapati Shivaji Maharaj International","city":"Mumbai","country":"India","icao":"VABB"},
    "BLR":{"name":"Kempegowda International","city":"Bengaluru","country":"India","icao":"VOBL"},
    "HYD":{"name":"Rajiv Gandhi International","city":"Hyderabad","country":"India","icao":"VOHS"},
    "MAA":{"name":"Chennai International","city":"Chennai","country":"India","icao":"VOMM"},
    "CCU":{"name":"Netaji Subhas Chandra Bose International","city":"Kolkata","country":"India","icao":"VECC"},
    "COK":{"name":"Cochin International","city":"Kochi","country":"India","icao":"VOCI"},
    "GOI":{"name":"Goa International (Dabolim)","city":"Goa","country":"India","icao":"VOGO"},
    "AMD":{"name":"Sardar Vallabhbhai Patel International","city":"Ahmedabad","country":"India","icao":"VAAH"},
    "JAI":{"name":"Jaipur International","city":"Jaipur","country":"India","icao":"VIJP"},
    "PNQ":{"name":"Pune Airport","city":"Pune","country":"India","icao":"VAPO"},
    "LKO":{"name":"Chaudhary Charan Singh International","city":"Lucknow","country":"India","icao":"VILK"},
    "ATQ":{"name":"Sri Guru Ram Dass Jee International","city":"Amritsar","country":"India","icao":"VIAR"},
    "TRV":{"name":"Trivandrum International","city":"Thiruvananthapuram","country":"India","icao":"VOTV"},
    "BBI":{"name":"Biju Patnaik International","city":"Bhubaneswar","country":"India","icao":"VEBS"},
    "NAG":{"name":"Dr. Babasaheb Ambedkar International","city":"Nagpur","country":"India","icao":"VANP"},
    "IXC":{"name":"Chandigarh International","city":"Chandigarh","country":"India","icao":"VICG"},
    "IXB":{"name":"Bagdogra Airport","city":"Siliguri","country":"India","icao":"VEBD"},
    "GAU":{"name":"Lokpriya Gopinath Bordoloi International","city":"Guwahati","country":"India","icao":"VEGT"},
    "PAT":{"name":"Jay Prakash Narayan International","city":"Patna","country":"India","icao":"VEPT"},
    "VNS":{"name":"Lal Bahadur Shastri International","city":"Varanasi","country":"India","icao":"VIBN"},
    "SXR":{"name":"Sheikh ul-Alam International","city":"Srinagar","country":"India","icao":"VISR"},
    "IXM":{"name":"Madurai Airport","city":"Madurai","country":"India","icao":"VOMD"},
    "CJB":{"name":"Coimbatore International","city":"Coimbatore","country":"India","icao":"VOCB"},
    "VTZ":{"name":"Visakhapatnam Airport","city":"Visakhapatnam","country":"India","icao":"VEVZ"},
    "BHO":{"name":"Raja Bhoj Airport","city":"Bhopal","country":"India","icao":"VABP"},
    "RPR":{"name":"Swami Vivekananda Airport","city":"Raipur","country":"India","icao":"VARP"},
    "IXR":{"name":"Birsa Munda Airport","city":"Ranchi","country":"India","icao":"VERC"},
    "DED":{"name":"Jolly Grant Airport","city":"Dehradun","country":"India","icao":"VIDN"},
    "AGR":{"name":"Agra Airport","city":"Agra","country":"India","icao":"VIAG"},
    # International
    "DXB":{"name":"Dubai International","city":"Dubai","country":"UAE","icao":"OMDB"},
    "LHR":{"name":"Heathrow","city":"London","country":"UK","icao":"EGLL"},
    "SIN":{"name":"Changi International","city":"Singapore","country":"Singapore","icao":"WSSS"},
    "NRT":{"name":"Narita International","city":"Tokyo","country":"Japan","icao":"RJAA"},
    "DOH":{"name":"Hamad International","city":"Doha","country":"Qatar","icao":"OTHH"},
    "CDG":{"name":"Charles de Gaulle","city":"Paris","country":"France","icao":"LFPG"},
    "HKG":{"name":"Hong Kong International","city":"Hong Kong","country":"HK","icao":"VHHH"},
    "SYD":{"name":"Kingsford Smith","city":"Sydney","country":"Australia","icao":"YSSY"},
    "JFK":{"name":"John F. Kennedy International","city":"New York","country":"USA","icao":"KJFK"},
    "LAX":{"name":"Los Angeles International","city":"Los Angeles","country":"USA","icao":"KLAX"},
    "KUL":{"name":"Kuala Lumpur International","city":"Kuala Lumpur","country":"Malaysia","icao":"WMKK"},
    "BKK":{"name":"Suvarnabhumi International","city":"Bangkok","country":"Thailand","icao":"VTBS"},
    "AUH":{"name":"Abu Dhabi International","city":"Abu Dhabi","country":"UAE","icao":"OMAA"},
    "FRA":{"name":"Frankfurt Airport","city":"Frankfurt","country":"Germany","icao":"EDDF"},
}

def get_all_airports() -> dict:
    return INDIAN_AIRPORTS

def search_airports(query: str, limit: int = 15) -> list:
    q = query.lower().strip()
    if not q or len(q) < 2:
        return []
    results = []
    # Exact IATA match first
    for code, info in INDIAN_AIRPORTS.items():
        if code.lower() == q:
            results.insert(0, {"iata": code, **info})
    # Then partial matches
    for code, info in INDIAN_AIRPORTS.items():
        if len(results) >= limit:
            break
        text = f"{code} {info['name']} {info['city']} {info['country']}".lower()
        if q in text and {"iata": code, **info} not in results:
            results.append({"iata": code, **info})
    return results[:limit]
