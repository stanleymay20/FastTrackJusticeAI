import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { Box, Typography, Paper, Grid, CircularProgress, Alert, Tooltip } from '@mui/material';
import { styled } from '@mui/material/styles';
import {
    WbSunny as SunIcon,
    AccessTime as TimeIcon,
    CalendarToday as CalendarIcon,
    RadioButtonChecked as PulseIcon,
    Lock as SealIcon,
    Brightness4 as NightIcon,
    Brightness7 as DayIcon,
    WbTwilight as TwilightIcon
} from '@mui/icons-material';

// Gate icons with their divine colors
const GATE_ICONS = {
    1: { icon: '🔥', color: '#FF4500', name: 'Fire of Beginning' },
    2: { icon: '💧', color: '#4169E1', name: 'River of Truth' },
    3: { icon: '📜', color: '#9370DB', name: 'Scroll of Mercy' },
    4: { icon: '💡', color: '#FFD700', name: 'Lamp of Justice' },
    5: { icon: '🔥', color: '#FF8C00', name: 'Flame of Knowledge' },
    6: { icon: '✨', color: '#E6E6FA', name: 'Mirror of Grace' },
    7: { icon: '🏛️', color: '#B8860B', name: 'Divine Architecture' }
};

// Phase colors
const PHASE_COLORS = {
    dawn: '#FFA500',
    noon: '#FFD700',
    dusk: '#FF4500',
    night: '#191970'
};

// Styled components
const DashboardContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  margin: theme.spacing(3),
  background: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.spacing(2),
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
  border: '1px solid rgba(255, 255, 255, 0.18)',
}));

const Card = styled(motion(Paper))(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  alignItems: 'center',
  background: 'rgba(255, 255, 255, 0.05)',
  backdropFilter: 'blur(5px)',
  borderRadius: theme.spacing(2),
  boxShadow: '0 4px 16px 0 rgba(31, 38, 135, 0.2)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
}));

const GateIcon = styled(motion.div)(({ theme, active, color }) => ({
  width: 60,
  height: 60,
  borderRadius: '50%',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  background: active ? `rgba(${color}, 0.2)` : 'rgba(255, 255, 255, 0.1)',
  border: `2px solid ${active ? color : 'rgba(255, 255, 255, 0.3)'}`,
  margin: theme.spacing(1),
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'scale(1.1)',
    background: active ? `rgba(${color}, 0.3)` : 'rgba(255, 255, 255, 0.2)',
  },
}));

const PulseRing = styled(motion.div)(({ theme, color }) => ({
  position: 'absolute',
  width: 200,
  height: 200,
  borderRadius: '50%',
  border: `2px solid ${color}`,
  boxShadow: `0 0 20px ${color}`,
}));

const PhaseRing = styled(motion.div)(({ theme, phase }) => ({
  position: 'absolute',
  width: 300,
  height: 300,
  borderRadius: '50%',
  border: `4px solid ${PHASE_COLORS[phase] || '#FFFFFF'}`,
  boxShadow: `0 0 30px ${PHASE_COLORS[phase] || '#FFFFFF'}`,
  opacity: 0.7,
}));

// Animation variants
const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

const gateIconVariants = {
  inactive: { scale: 1, opacity: 0.7 },
  active: { scale: 1.1, opacity: 1 },
  hover: { scale: 1.2, opacity: 1 },
};

const pulseRingVariants = {
  initial: { scale: 0.8, opacity: 0.5 },
  animate: (pulse) => ({ 
    scale: [0.8, 1.2, 0.8],
    opacity: [0.5, 0.8, 0.5],
    transition: {
      duration: 3,
      repeat: Infinity,
      ease: "easeInOut",
      delay: (pulse % 7) * 0.2 // Stagger animation based on pulse
    }
  }),
};

const phaseRingVariants = {
  initial: { scale: 0.9, opacity: 0.5 },
  animate: { 
    scale: [0.9, 1.1, 0.9],
    opacity: [0.5, 0.7, 0.5],
    transition: {
      duration: 5,
      repeat: Infinity,
      ease: "easeInOut"
    }
  },
};

const ScrollDashboard = () => {
  const [scrollData, setScrollData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [location, setLocation] = useState(null);
  const [locationError, setLocationError] = useState(null);

  // Get user's location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          });
        },
        (error) => {
          console.error('Error getting location:', error);
          setLocationError('Unable to get your location. Using default calculations.');
        }
      );
    } else {
      setLocationError('Geolocation is not supported by your browser. Using default calculations.');
    }
  }, []);

  // Fetch scroll data
  useEffect(() => {
    const fetchScrollData = async () => {
      try {
        setLoading(true);
        const params = location 
          ? { latitude: location.latitude, longitude: location.longitude }
          : {};
        
        const response = await axios.get('http://localhost:8000/api/scroll-time', { params });
        setScrollData(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching scroll data:', err);
        setError('Failed to fetch scroll data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchScrollData();
    const interval = setInterval(fetchScrollData, 60000); // Update every minute

    return () => clearInterval(interval);
  }, [location]);

  // Get phase icon
  const getPhaseIcon = (phase) => {
    switch(phase) {
      case 'dawn':
        return <TwilightIcon fontSize="large" sx={{ color: PHASE_COLORS.dawn }} />;
      case 'noon':
        return <DayIcon fontSize="large" sx={{ color: PHASE_COLORS.noon }} />;
      case 'dusk':
        return <TwilightIcon fontSize="large" sx={{ color: PHASE_COLORS.dusk }} />;
      case 'night':
        return <NightIcon fontSize="large" sx={{ color: PHASE_COLORS.night }} />;
      default:
        return <SunIcon fontSize="large" />;
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box m={3}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <DashboardContainer>
      {locationError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {locationError}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <AnimatePresence mode="wait">
            <Card
              key={scrollData?.phase}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              transition={{ duration: 0.5 }}
            >
              <Box position="relative" display="flex" justifyContent="center" alignItems="center" height={200}>
                <PhaseRing
                  phase={scrollData?.phase}
                  variants={phaseRingVariants}
                  initial="initial"
                  animate="animate"
                />
                <Box display="flex" flexDirection="column" alignItems="center" sx={{ position: 'relative', zIndex: 1 }}>
                  {getPhaseIcon(scrollData?.phase)}
                  <Typography variant="h4" gutterBottom>
                    {scrollData?.phase}
                  </Typography>
                </Box>
              </Box>
              <Typography variant="h6" color="textSecondary">
                Time Remaining: {scrollData?.time_remaining}
              </Typography>
              <Typography variant="body1">
                Next Phase: {scrollData?.next_phase}
              </Typography>
              {scrollData?.is_active && (
                <Box mt={2} p={1} bgcolor="rgba(255, 215, 0, 0.2)" borderRadius={1}>
                  <Typography variant="body2" color="gold">
                    ✨ Scroll is currently active ✨
                  </Typography>
                </Box>
              )}
            </Card>
          </AnimatePresence>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Box position="relative" display="flex" justifyContent="center" alignItems="center" height={200}>
              <PulseRing
                color={scrollData?.is_active ? 'rgba(255, 215, 0, 0.5)' : 'rgba(255, 255, 255, 0.3)'}
                variants={pulseRingVariants}
                initial="initial"
                animate="animate"
                custom={scrollData?.enano_pulse}
              />
              <Typography variant="h3" sx={{ position: 'relative', zIndex: 1 }}>
                {scrollData?.solar_hour}
              </Typography>
            </Box>
            <Typography variant="h6">Solar Hour</Typography>
          </Card>
        </Grid>
        
        <Grid item xs={12}>
          <Card
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Typography variant="h5" gutterBottom>
              Gates
            </Typography>
            <Box display="flex" justifyContent="center" flexWrap="wrap">
              {[1, 2, 3, 4, 5, 6, 7].map((gate) => {
                const gateColor = GATE_ICONS[gate]?.color || '#FFFFFF';
                return (
                  <Tooltip key={gate} title={GATE_ICONS[gate]?.name || `Gate ${gate}`}>
                    <GateIcon
                      active={gate === scrollData?.gate}
                      color={gateColor}
                      variants={gateIconVariants}
                      initial="inactive"
                      animate={gate === scrollData?.gate ? "active" : "inactive"}
                      whileHover="hover"
                      transition={{ duration: 0.3 }}
                    >
                      <Typography variant="h6">{gate}</Typography>
                    </GateIcon>
                  </Tooltip>
                );
              })}
            </Box>
            {scrollData?.gate_name && (
              <Typography variant="body1" mt={2} textAlign="center">
                {scrollData.gate_name}
              </Typography>
            )}
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <Box position="relative" display="flex" justifyContent="center" alignItems="center" height={200}>
              <PulseRing
                color={`rgba(${scrollData?.enano_pulse % 7 === 0 ? '255, 215, 0' : '255, 255, 255'}, 0.5)`}
                variants={pulseRingVariants}
                initial="initial"
                animate="animate"
                custom={scrollData?.enano_pulse}
              />
              <Typography variant="h3" sx={{ position: 'relative', zIndex: 1 }}>
                {scrollData?.enano_pulse}
              </Typography>
            </Box>
            <Typography variant="h6">ENano Pulse</Typography>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            <Typography variant="h4" gutterBottom>
              {scrollData?.time_remaining_today}
            </Typography>
            <Typography variant="h6">Time Remaining Today</Typography>
          </Card>
        </Grid>
        
        {scrollData?.sunrise && scrollData?.sunset && (
          <Grid item xs={12}>
            <Card
              variants={cardVariants}
              initial="hidden"
              animate="visible"
              transition={{ duration: 0.5, delay: 1 }}
            >
              <Typography variant="h5" gutterBottom>
                Solar Times
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Box p={2} bgcolor="rgba(255, 165, 0, 0.1)" borderRadius={1}>
                    <Typography variant="subtitle1">Dawn</Typography>
                    <Typography variant="body2">{new Date(scrollData.dawn).toLocaleTimeString()}</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box p={2} bgcolor="rgba(255, 215, 0, 0.1)" borderRadius={1}>
                    <Typography variant="subtitle1">Sunrise</Typography>
                    <Typography variant="body2">{new Date(scrollData.sunrise).toLocaleTimeString()}</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box p={2} bgcolor="rgba(255, 69, 0, 0.1)" borderRadius={1}>
                    <Typography variant="subtitle1">Sunset</Typography>
                    <Typography variant="body2">{new Date(scrollData.sunset).toLocaleTimeString()}</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box p={2} bgcolor="rgba(75, 0, 130, 0.1)" borderRadius={1}>
                    <Typography variant="subtitle1">Dusk</Typography>
                    <Typography variant="body2">{new Date(scrollData.dusk).toLocaleTimeString()}</Typography>
                  </Box>
                </Grid>
              </Grid>
            </Card>
          </Grid>
        )}
      </Grid>
    </DashboardContainer>
  );
};

export default ScrollDashboard; 