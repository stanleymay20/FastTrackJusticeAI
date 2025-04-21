import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Divider,
  Button,
  Alert,
} from '@mui/material';
import {
  Description as DescriptionIcon,
  Category as CategoryIcon,
  Gavel as GavelIcon,
} from '@mui/icons-material';

const Results = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const storedResults = localStorage.getItem('caseResults');
    if (!storedResults) {
      setError('No results found. Please upload a case first.');
      return;
    }

    try {
      setResults(JSON.parse(storedResults));
    } catch (err) {
      setError('Error loading results. Please try again.');
    }
  }, []);

  const handleNewCase = () => {
    navigate('/upload');
  };

  if (error) {
    return (
      <Box>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={handleNewCase}>
          Upload New Case
        </Button>
      </Box>
    );
  }

  if (!results) {
    return null;
  }

  const { summary, classification, judgment_draft, confidence_scores } = results;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Case Analysis Results
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Section */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <DescriptionIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Case Summary</Typography>
              </Box>
              <Typography variant="body1" paragraph>
                {summary}
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Confidence Score
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={confidence_scores.summary * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {Math.round(confidence_scores.summary * 100)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Classification Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CategoryIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Case Classification</Typography>
              </Box>
              <Box sx={{ mb: 2 }}>
                {Object.entries(classification)
                  .filter(([key]) => key !== 'confidence')
                  .map(([category, score]) => (
                    <Chip
                      key={category}
                      label={`${category}: ${Math.round(score * 100)}%`}
                      sx={{ m: 0.5 }}
                      color={score > 0.5 ? 'primary' : 'default'}
                    />
                  ))}
              </Box>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Classification Confidence
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={confidence_scores.classification * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {Math.round(confidence_scores.classification * 100)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Judgment Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <GavelIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Generated Judgment</Typography>
              </Box>
              <Paper
                variant="outlined"
                sx={{
                  p: 2,
                  maxHeight: 300,
                  overflow: 'auto',
                  backgroundColor: 'background.default',
                }}
              >
                <Typography variant="body2" style={{ whiteSpace: 'pre-line' }}>
                  {judgment_draft}
                </Typography>
              </Paper>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Judgment Confidence
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={confidence_scores.judgment * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {Math.round(confidence_scores.judgment * 100)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Overall Confidence */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Overall Analysis Confidence
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Box sx={{ flexGrow: 1, mr: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={confidence_scores.overall * 100}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {Math.round(confidence_scores.overall * 100)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
        <Button variant="outlined" onClick={handleNewCase}>
          Analyze Another Case
        </Button>
        <Button
          variant="contained"
          onClick={() => {
            // Implement download functionality
            const blob = new Blob([JSON.stringify(results, null, 2)], {
              type: 'application/json',
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'case-analysis-results.json';
            a.click();
            URL.revokeObjectURL(url);
          }}
        >
          Download Results
        </Button>
      </Box>
    </Box>
  );
};

export default Results; 