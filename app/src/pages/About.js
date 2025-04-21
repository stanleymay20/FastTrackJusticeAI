import React from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
} from '@mui/material';
import {
  Code as CodeIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Psychology as PsychologyIcon,
  Build as BuildIcon,
} from '@mui/icons-material';

const About = () => {
  const technicalDetails = [
    {
      icon: <PsychologyIcon />,
      title: 'AI Models',
      description: 'Built with state-of-the-art transformer models including BART, BERT, and T5.',
    },
    {
      icon: <CodeIcon />,
      title: 'Technology Stack',
      description: 'React, FastAPI, PyTorch, and Hugging Face Transformers.',
    },
    {
      icon: <SecurityIcon />,
      title: 'Security',
      description: 'End-to-end encryption and secure data processing.',
    },
    {
      icon: <SpeedIcon />,
      title: 'Performance',
      description: 'Optimized for fast processing with GPU acceleration support.',
    },
    {
      icon: <StorageIcon />,
      title: 'Data Handling',
      description: 'Support for multiple document formats and efficient data processing.',
    },
    {
      icon: <BuildIcon />,
      title: 'Deployment',
      description: 'Docker-ready with cloud deployment options.',
    },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          About FastTrackJusticeAI
        </Typography>

        <Typography variant="body1" paragraph>
          FastTrackJusticeAI is an open-source platform that leverages artificial intelligence
          to streamline legal case processing. Our mission is to make legal analysis more
          efficient, accessible, and accurate through cutting-edge AI technology.
        </Typography>

        <Grid container spacing={4} sx={{ mt: 2 }}>
          {/* Technical Details */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Technical Architecture
                </Typography>
                <Grid container spacing={2}>
                  {technicalDetails.map((detail, index) => (
                    <Grid item xs={12} sm={6} key={index}>
                      <Paper
                        sx={{
                          p: 2,
                          height: '100%',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          textAlign: 'center',
                        }}
                      >
                        <Box sx={{ color: 'primary.main', mb: 1 }}>
                          {detail.icon}
                        </Box>
                        <Typography variant="h6" gutterBottom>
                          {detail.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {detail.description}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Key Features */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Key Features
                </Typography>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <PsychologyIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Legal Case Summarization"
                      secondary="Generate concise summaries using advanced AI models"
                    />
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemIcon>
                      <PsychologyIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Case Classification"
                      secondary="Intelligent categorization of legal cases"
                    />
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemIcon>
                      <PsychologyIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Judgment Drafting"
                      secondary="AI-assisted legal opinion generation"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Project Vision */}
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Project Vision
          </Typography>
          <Typography variant="body1" paragraph>
            Our vision is to revolutionize the legal industry by making advanced AI
            technology accessible to legal professionals worldwide. We believe in
            transparency, collaboration, and continuous improvement through open-source
            development.
          </Typography>
        </Box>

        {/* Performance Metrics */}
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Performance Metrics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary">
                  89%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Classification Accuracy
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary">
                  42%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  ROUGE Score
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary">
                  85%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  BERTScore
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Box>
    </Container>
  );
};

export default About; 