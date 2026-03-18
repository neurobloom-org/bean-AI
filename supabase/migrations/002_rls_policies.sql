-- Enable Row Level Security
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodic_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE semantic_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE procedural_memory ENABLE ROW LEVEL SECURITY;

-- Create Policies
CREATE POLICY "Users can only see their own profile" 
ON user_profiles FOR ALL USING (auth.uid() = id);

CREATE POLICY "Users can only see their own episodic memories" 
ON episodic_memory FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own semantic memories" 
ON semantic_memory FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own procedural memories" 
ON procedural_memory FOR ALL USING (auth.uid() = user_id);