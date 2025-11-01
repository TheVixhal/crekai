-- Add Learn PyTorch project
INSERT INTO projects (slug, title, description, total_steps) VALUES
('learn-pytorch', 'Learn PyTorch', 'Deep learning with PyTorch for neural networks', 1)
ON CONFLICT (slug) DO NOTHING;
