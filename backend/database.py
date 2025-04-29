from pymongo import MongoClient, WriteConcern
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
from dotenv import load_dotenv
import os
import sys

load_dotenv()

def test_db_connection():
    try:
        # MongoDB connection with proper timeouts
        MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        print(f"Attempting to connect to MongoDB at: {MONGO_URI}")
        
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,  # 5 second timeout for server selection
            connectTimeoutMS=5000,          # 5 second timeout for initial connection
            socketTimeoutMS=5000,           # 5 second timeout for socket operations
            maxPoolSize=50,                 # Maximum number of connections
            retryWrites=True                # Enable retrying write operations
        )
        
        # Test the connection
        client.server_info()
        print("✅ Successfully connected to MongoDB server")
        
        # Get database
        db = client.ai_fitness_tracker
        print(f"✅ Connected to database: {db.name}")
        
        # Test database write permission with write concern
        write_concern = WriteConcern(w=1, wtimeout=5000)
        test_collection = db.get_collection('connection_test', write_concern=write_concern)
        test_result = test_collection.insert_one({"test": "connection"})
        print("✅ Successfully wrote to database")
        
        test_collection.delete_one({"_id": test_result.inserted_id})
        print("✅ Successfully deleted test document")
        
        return client, db
    except ServerSelectionTimeoutError as e:
        print(f"❌ MongoDB server selection timeout: {str(e)}")
        print("Please make sure MongoDB is running and accessible")
        sys.exit(1)
    except OperationFailure as e:
        print(f"❌ MongoDB operation failure: {str(e)}")
        print("Please check MongoDB permissions and configuration")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {str(e)}")
        print("Please make sure MongoDB is running and accessible")
        sys.exit(1)

# Initialize database connection
print("\n=== Initializing Database Connection ===")
client, db = test_db_connection()

# Collections with write concern
write_concern = WriteConcern(w=1, wtimeout=5000)
users = db.get_collection('users', write_concern=write_concern)
exercises = db.get_collection('exercises', write_concern=write_concern)
workouts = db.get_collection('workouts', write_concern=write_concern)
user_profiles = db.get_collection('user_profiles', write_concern=write_concern)
exercise_plans = db.get_collection('exercise_plans', write_concern=write_concern)
ai_models = db.get_collection('ai_models', write_concern=write_concern)

# Create indexes
try:
    print("\n=== Creating Database Indexes ===")
    users.create_index("username", unique=True)
    users.create_index("email", unique=True)
    exercises.create_index([("username", 1), ("timestamp", -1)])
    workouts.create_index([("username", 1), ("date", -1)])
    user_profiles.create_index("username", unique=True)
    exercise_plans.create_index([("username", 1), ("created_at", -1)])
    print("✅ Database indexes created successfully")
except Exception as e:
    print(f"❌ Error creating database indexes: {str(e)}")
    sys.exit(1)

# Print collection information
print("\n=== Database Collections ===")
for collection in [users, exercises, workouts, user_profiles, exercise_plans, ai_models]:
    print(f"- {collection.name}: {collection.count_documents({})} documents")

async def init_db():
    try:
        # Test database connection
        await test_db_connection()
        
        # Initialize collections with write concern
        write_concern = WriteConcern(w=1, wtimeout=5000)
        
        users = db.get_collection('users', write_concern=write_concern)
        exercises = db.get_collection('exercises', write_concern=write_concern)
        workouts = db.get_collection('workouts', write_concern=write_concern)
        user_profiles = db.get_collection('user_profiles', write_concern=write_concern)
        exercise_plans = db.get_collection('exercise_plans', write_concern=write_concern)
        ai_models = db.get_collection('ai_models', write_concern=write_concern)
        
        # Create indexes
        await users.create_index([("username", 1)], unique=True)
        await users.create_index([("email", 1)], unique=True)
        await exercises.create_index([("name", 1)], unique=True)
        await workouts.create_index([("user_id", 1), ("date", -1)])  # Index for workout history queries
        await exercise_plans.create_index([("user_id", 1), ("name", 1)])
        
        # Print collection stats
        print(f"Users: {await users.count_documents({})}")
        print(f"Exercises: {await exercises.count_documents({})}")
        print(f"Workouts: {await workouts.count_documents({})}")
        print(f"User Profiles: {await user_profiles.count_documents({})}")
        print(f"Exercise Plans: {await exercise_plans.count_documents({})}")
        print(f"AI Models: {await ai_models.count_documents({})}")
        
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise 