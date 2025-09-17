-- Quick check if MRI predictions table exists
SELECT EXISTS (
   SELECT FROM information_schema.tables 
   WHERE table_schema = 'public' 
   AND table_name = 'mri_predictions'
) as table_exists;
