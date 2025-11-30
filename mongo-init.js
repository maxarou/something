db = db.getSiblingDB('robotdb')

db.createCollection('visited_cells')
db.createCollection('episodes')
db.createCollection('robots')

print("Mongo initialized for multi-robot exploration.");
