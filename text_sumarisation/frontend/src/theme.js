export const colors = {
  primary: '#722F37',      // Wine - đỏ rượu vang chủ đạo
  primaryDark: '#5a252c',  // Wine đậm hơn cho hover/gradient
  primaryLight: '#8B4550', // Wine nhạt hơn
  accent: '#e8d5b7',        // Champagne - vàng nhấn
  accentDark: '#d4c4a8',     // Champagne đậm
  header: '#faf8f5',        // Ivory - trắng ngà header
  bg: '#f5f2ed',            // Cream - nền chính
  bgDark: '#2d2426',        // Dark wine - nền hero/footer
  text: '#2d2d2d',          // Text chính
  textLight: '#6b5b5e',     // Text phụ
  border: '#e8e4e0',        // Border
  white: '#ffffff',
};

export const gradients = {
  hero: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 50%, #3d1f24 100%)`,
  card: `linear-gradient(145deg, ${colors.white} 0%, ${colors.header} 100%)`,
};