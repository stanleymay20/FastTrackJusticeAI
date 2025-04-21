import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Container,
} from '@mui/material';
import {
  Description as DescriptionIcon,
  Category as CategoryIcon,
  Gavel as GavelIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  Code as CodeIcon,
} from '@mui/icons-material';

const features = [
  {
    icon: <DescriptionIcon fontSize="large" />,
    title: 'Legal Case Summarization',
    description: 'Automatically generate concise summaries of legal cases using advanced AI models.',
  },
  {
    icon: <CategoryIcon fontSize="large" />,
    title: 'Intelligent Classification',
    description: 'Classify cases by type and urgency using multi-label classification.',
  },
  {
    icon: <GavelIcon fontSize="large" />,
    title: 'Judgment Drafting',
    description: 'Generate structured legal opinions based on case analysis.',
  },
  {
    icon: <SpeedIcon fontSize="large" />,
    title: 'Fast Processing',
    description: 'Process legal documents quickly and efficiently.',
  },
  {
    icon: <SecurityIcon fontSize="large" />,
    title: 'Secure & Private',
    description: 'Your data is processed securely and remains private.',
  },
  {
    icon: <CodeIcon fontSize="large" />,
    title: 'Open Source',
    description: 'Built with open-source technologies for transparency and collaboration.',
  },
];

const Home = () => {
  const navigate = useNavigate();

  return (
    <Box>
      {/* Hero Section */}
      <Box
        sx={{
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          py: 8,
          mb: 6,
          borderRadius: 2,
        }}
      >
        <Container maxWidth="md">
          <Typography variant="h2" component="h1" gutterBottom>
            FastTrackJusticeAI
          </Typography>
          <Typography variant="h5" component="h2" gutterBottom>
            Justice, Accelerated by Intelligence
          </Typography>
          <Typography variant="body1" paragraph sx={{ mb: 4 }}>
            Transform legal case processing with AI-powered analysis, summarization,
            and judgment drafting.
          </Typography>
          <Button
            variant="contained"
            color="secondary"
            size="large"
            onClick={() => navigate('/upload')}
            sx={{ mr: 2 }}
          >
            Get Started
          </Button>
          <Button
            variant="outlined"
            color="inherit"
            size="large"
            onClick={() => navigate('/about')}
          >
            Learn More
          </Button>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg">
        <Typography variant="h4" component="h2" gutterBottom align="center">
          Features
        </Typography>
        <Grid container spacing={4} sx={{ mb: 6 }}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    transition: 'transform 0.2s ease-in-out',
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'center',
                      mb: 2,
                      color: 'primary.main',
                    }}
                  >
                    {feature.icon}
                  </Box>
                  <Typography variant="h6" component="h3" gutterBottom align="center">
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" align="center">
                    {feature.description}
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: 'center', pb: 2 }}>
                  <Button
                    size="small"
                    color="primary"
                    onClick={() => navigate('/upload')}
                  >
                    Try Now
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Call to Action */}
        <Box
          sx={{
            textAlign: 'center',
            py: 6,
            bgcolor: 'background.paper',
            borderRadius: 2,
          }}
        >
          <Typography variant="h4" component="h2" gutterBottom>
            Ready to Transform Your Legal Workflow?
          </Typography>
          <Typography variant="body1" paragraph sx={{ mb: 4 }}>
            Join the future of legal technology with FastTrackJusticeAI.
          </Typography>
          <Button
            variant="contained"
            color="primary"
            size="large"
            onClick={() => navigate('/upload')}
          >
            Start Processing Cases
          </Button>
        </Box>
      </Container>
    </Box>
  );
};

export default Home; 