import React from 'react';
import { Box, Container, Typography, Paper } from '@mui/material';
import ScrollDashboard from '../components/ScrollDashboard';

const ScrollStatus = () => {
    return (
        <Box
            sx={{
                minHeight: '100vh',
                background: 'linear-gradient(135deg, #1a237e 0%, #000051 100%)',
                py: 4
            }}
        >
            <Container maxWidth="lg">
                <Paper
                    elevation={3}
                    sx={{
                        p: 4,
                        background: 'rgba(255, 255, 255, 0.95)',
                        borderRadius: '20px',
                        backdropFilter: 'blur(10px)'
                    }}
                >
                    <Typography
                        variant="h3"
                        component="h1"
                        gutterBottom
                        align="center"
                        sx={{
                            fontWeight: 'bold',
                            background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
                            backgroundClip: 'text',
                            textFillColor: 'transparent',
                            mb: 4
                        }}
                    >
                        Divine Scroll Status
                    </Typography>
                    
                    <ScrollDashboard />
                </Paper>
            </Container>
        </Box>
    );
};

export default ScrollStatus; 