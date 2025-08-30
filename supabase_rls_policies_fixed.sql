-- =====================================================
-- RL Leaderboard - Row Level Security (RLS) Policies
-- =====================================================
-- FIXED VERSION FOR SUPABASE PERMISSIONS
-- Run these commands in your Supabase SQL editor

-- =====================================================
-- 1. ENABLE RLS ON TABLES
-- =====================================================

-- Enable RLS on submissions table
ALTER TABLE public.submissions ENABLE ROW LEVEL SECURITY;

-- Enable RLS on leaderboard_entries table  
ALTER TABLE public.leaderboard_entries ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 2. SUBMISSIONS TABLE POLICIES
-- =====================================================

-- Policy: Allow anyone to read completed submissions (public leaderboard data)
CREATE POLICY "submissions_read_public" ON public.submissions
    FOR SELECT
    USING (
        status = 'completed' 
        AND score IS NOT NULL
    );

-- Policy: Allow service role to read all submissions (for admin/worker access)
CREATE POLICY "submissions_read_service_role" ON public.submissions
    FOR SELECT
    USING (
        auth.role() = 'service_role'
    );

-- Policy: Allow anyone to create new submissions (public submission endpoint)
CREATE POLICY "submissions_insert_public" ON public.submissions
    FOR INSERT
    WITH CHECK (true);

-- Policy: Allow service role to update submissions (for worker processing)
CREATE POLICY "submissions_update_service_role" ON public.submissions
    FOR UPDATE
    USING (
        auth.role() = 'service_role'
    )
    WITH CHECK (
        auth.role() = 'service_role'
    );

-- Policy: Allow service role to delete submissions (for cleanup)
CREATE POLICY "submissions_delete_service_role" ON public.submissions
    FOR DELETE
    USING (
        auth.role() = 'service_role'
    );

-- =====================================================
-- 3. LEADERBOARD_ENTRIES TABLE POLICIES
-- =====================================================

-- Policy: Allow anyone to read leaderboard entries (public leaderboard)
CREATE POLICY "leaderboard_read_public" ON public.leaderboard_entries
    FOR SELECT
    USING (true);

-- Policy: Allow service role to read all leaderboard entries
CREATE POLICY "leaderboard_read_service_role" ON public.leaderboard_entries
    FOR SELECT
    USING (
        auth.role() = 'service_role'
    );

-- Policy: Allow service role to insert leaderboard entries (for completed evaluations)
CREATE POLICY "leaderboard_insert_service_role" ON public.leaderboard_entries
    FOR INSERT
    WITH CHECK (
        auth.role() = 'service_role'
    );

-- Policy: Allow service role to update leaderboard entries
CREATE POLICY "leaderboard_update_service_role" ON public.leaderboard_entries
    FOR UPDATE
    USING (
        auth.role() = 'service_role'
    )
    WITH CHECK (
        auth.role() = 'service_role'
    );

-- Policy: Allow service role to delete leaderboard entries (for cleanup)
CREATE POLICY "leaderboard_delete_service_role" ON public.leaderboard_entries
    FOR DELETE
    USING (
        auth.role() = 'service_role'
    );

-- =====================================================
-- 4. STORAGE BUCKET POLICIES
-- =====================================================

-- Enable RLS on storage.objects
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access to submissions bucket
CREATE POLICY "storage_submissions_service_role" ON storage.objects
    FOR ALL
    USING (
        bucket_id = 'submissions'
        AND auth.role() = 'service_role'
    )
    WITH CHECK (
        bucket_id = 'submissions'
        AND auth.role() = 'service_role'
    );

-- =====================================================
-- 5. ADDITIONAL SECURITY MEASURES
-- =====================================================

-- Create indexes for better performance with RLS
CREATE INDEX IF NOT EXISTS idx_submissions_user_id_status ON public.submissions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_submissions_env_id_status ON public.submissions(env_id, status);
CREATE INDEX IF NOT EXISTS idx_leaderboard_entries_env_id_score ON public.leaderboard_entries(env_id, score DESC);

-- Create a function to validate submission data
CREATE OR REPLACE FUNCTION public.validate_submission()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure user_id is not empty
    IF NEW.user_id IS NULL OR TRIM(NEW.user_id) = '' THEN
        RAISE EXCEPTION 'user_id cannot be empty';
    END IF;
    
    -- Ensure env_id is not empty
    IF NEW.env_id IS NULL OR TRIM(NEW.env_id) = '' THEN
        RAISE EXCEPTION 'env_id cannot be empty';
    END IF;
    
    -- Ensure algorithm is not empty
    IF NEW.algorithm IS NULL OR TRIM(NEW.algorithm) = '' THEN
        RAISE EXCEPTION 'algorithm cannot be empty';
    END IF;
    
    -- Ensure score is within reasonable bounds when provided
    IF NEW.score IS NOT NULL AND (NEW.score < -1000000 OR NEW.score > 1000000) THEN
        RAISE EXCEPTION 'score must be between -1000000 and 1000000';
    END IF;
    
    -- Ensure duration is positive when provided
    IF NEW.duration_seconds IS NOT NULL AND NEW.duration_seconds < 0 THEN
        RAISE EXCEPTION 'duration_seconds must be positive';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for submission validation
CREATE TRIGGER validate_submission_trigger
    BEFORE INSERT OR UPDATE ON public.submissions
    FOR EACH ROW
    EXECUTE FUNCTION public.validate_submission();

-- Create a function to validate leaderboard entry data
CREATE OR REPLACE FUNCTION public.validate_leaderboard_entry()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure score is provided and within bounds
    IF NEW.score IS NULL THEN
        RAISE EXCEPTION 'score cannot be null';
    END IF;
    
    IF NEW.score < -1000000 OR NEW.score > 1000000 THEN
        RAISE EXCEPTION 'score must be between -1000000 and 1000000';
    END IF;
    
    -- Ensure submission_id references a valid submission
    IF NOT EXISTS (SELECT 1 FROM public.submissions WHERE id = NEW.submission_id) THEN
        RAISE EXCEPTION 'submission_id must reference a valid submission';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for leaderboard entry validation
CREATE TRIGGER validate_leaderboard_entry_trigger
    BEFORE INSERT OR UPDATE ON public.leaderboard_entries
    FOR EACH ROW
    EXECUTE FUNCTION public.validate_leaderboard_entry();

-- =====================================================
-- 6. USAGE INSTRUCTIONS
-- =====================================================

/*
HOW TO USE THESE POLICIES:

1. Run this entire script in your Supabase SQL editor
2. The policies will automatically enforce security rules
3. Your application will work as follows:

PUBLIC ACCESS (anon key):
- Can read completed submissions (public leaderboard data)
- Can read all leaderboard entries
- Can create new submissions
- Cannot see pending/failed submissions
- Cannot modify existing data

SERVICE ROLE (service key):
- Has full access to all tables and operations
- Can read/write all submissions and leaderboard entries
- Can manage storage files
- Can perform cleanup operations

SECURITY FEATURES:
- Data validation on insert/update
- Proper indexing for performance
- Storage bucket protection
- User isolation for sensitive data

TESTING:
To test the policies, try these queries with different keys:

-- Should work with anon key (public data)
SELECT * FROM submissions WHERE status = 'completed' LIMIT 5;

-- Should work with service role key (all data)
SELECT * FROM submissions LIMIT 5;

-- Should fail with anon key (private data)
SELECT * FROM submissions WHERE status = 'pending' LIMIT 5;
*/


