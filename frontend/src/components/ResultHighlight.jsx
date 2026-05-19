const ResultHighlight = ({ text, keyword }) => {
  if (!keyword || keyword.trim() === '') {
    return <span>{text}</span>;
  }

  const escapedKeyword = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const parts = text.split(new RegExp(`(${escapedKeyword})`, 'gi'));

  return (
    <span>
      {parts.map((part, i) => 
        part.toLowerCase() === keyword.toLowerCase() ? (
          <mark key={i} className="bg-[#f4d03f] text-[#2d2d2d] px-0.5 rounded font-semibold">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </span>
  );
};

export default ResultHighlight;