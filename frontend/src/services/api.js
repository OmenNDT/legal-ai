import axios from 'axios';

//Instance axios dùng chung
export const api = axios.create({
  baseURL: 'api',
  headers: {
    'Content-Type': 'application/json',
  },
});


//Mock data
const MOCK_LAWS = [
  {
    id: 'hien-phap-2013',
    title: 'Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam 2013',
    type: 'Hiến pháp',
    year: 2013,
    content: `Điều 1. Nước Cộng hòa xã hội chủ nghĩa Việt Nam là một nước độc lập, có chủ quyền, thống nhất và toàn vẹn lãnh thổ.
Điều 2. Nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam là nhà nước pháp quyền xã hội chủ nghĩa của Nhân dân, do Nhân dân, vì Nhân dân.
Điều 3. Quyền lực nhà nước là thống nhất, có sự phân công, phối hợp và kiểm soát giữa các cơ quan nhà nước trong việc thực hiện các quyền lập pháp, hành pháp, tư pháp.`
  },
  {
    id: 'dan-su-2015',
    title: 'Bộ luật Dân sự 2015',
    type: 'Bộ luật',
    year: 2015,
    content: `Điều 1. Mục đích của Bộ luật Dân sự
Bộ luật này quy định về địa vị pháp lý của cá nhân, pháp nhân; quyền nhân thân, quyền tài sản; quan hệ dân sự.
Điều 2. Giải thích từ ngữ
Trong Bộ luật này, các từ ngữ dưới đây được hiểu như sau:
1. Dân sự là các quan hệ phát sinh giữa các chủ thể bình đẳng, độc lập, tự do ý chí.`
  },
  {
    id: 'hinh-su-2015',
    title: 'Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017)',
    type: 'Bộ luật',
    year: 2017,
    content: `Điều 1. Nhiệm vụ của Bộ luật Hình sự
Bộ luật Hình sự quy định về tội phạm và hình phạt, quy định về các biện pháp xử lý vi phạm hành chính.
Điều 2. Nguyên tắc xử lý hình sự
1. Mọi hành vi truy cứu trách nhiệm hình sự phải có căn cứ vào Bộ luật này.`
  },
  {
    id: 'lao-dong-2019',
    title: 'Bộ luật Lao động 2019',
    type: 'Bộ luật',
    year: 2019,
    content: `Điều 1. Phạm vi điều chỉnh
Bộ luật này quy định về lao động; quan hệ lao động; quan hệ có liên quan đến quan hệ lao động.
Điều 2. Đối tượng áp dụng
1. Người lao động, người sử dụng lao động.`
  },
  {
    id: 'dat-dai-2013',
    title: 'Luật Đất đai 2013',
    type: 'Luật',
    year: 2013,
    content: `Điều 1. Phạm vi điều chỉnh
Luật này quy định về chế độ sở hữu đất đai, quyền hạn và trách nhiệm của Nhà nước đại diện chủ sở hữu toàn dân về đất đai.
Điều 2. Đối tượng áp dụng
Luật này áp dụng đối với tổ chức, cá nhân trong nước, người Việt Nam định cư ở nước ngoài.`
  }
];

const MOCK_SUMMARIES = {
  'hien-phap-2013': 'Hiến pháp 2013 là văn bản pháp lý cao nhất của Việt Nam, gồm 11 chương và 120 điều. Văn bản khẳng định chế độ xã hội chủ nghĩa, quyền làm chủ của nhân dân, và nguyên tắc tập quyền.',
  'dan-su-2015': 'Bộ luật Dân sự 2015 gồm 6 phần, 689 điều. Quy định về chủ thể, quyền nhân thân, giao dịch dân sự, quyền sở hữu, nghĩa vụ và hợp đồng.',
  'hinh-su-2015': 'Bộ luật Hình sự 2015 (sửa đổi 2017) gồm 29 chương, 426 điều. Quy định tội phạm, hình phạt, biện pháp tư pháp và nguyên tắc xử lý hình sự.',
  'lao-dong-2019': 'Bộ luật Lao động 2019 gồm 17 chương, 220 điều. Quy định về hợp đồng lao động, tiền lương, thời giờ làm việc, an toàn lao động.',
  'dat-dai-2013': 'Luật Đất đai 2013 gồm 14 chương, 212 điều. Quy định về quy hoạch, kế hoạch sử dụng đất, thu hồi đất, giá đất và quyền của người sử dụng đất.'
};

const MOCK_QA = [
  {
    keywords: ['hình sự', 'tội phạm', 'phạt tù', 'án'],
    answer: 'Theo Bộ luật Hình sự 2015, tội phạm là hành vi nguy hiểm cho xã hội được quy định trong Bộ luật Hình sự, do người có năng lực trách nhiệm hình sự thực hiện một cách cố ý hoặc vô ý.'
  },
  {
    keywords: ['dân sự', 'hợp đồng', 'sở hữu', 'tài sản'],
    answer: 'Bộ luật Dân sự 2015 quy định các quan hệ dân sự giữa các chủ thể bình đẳng. Hợp đồng dân sự phải tuân thủ nguyên tắc tự do thỏa thuận, bình đẳng và thiện chí.'
  },
  {
    keywords: ['lao động', 'hợp đồng lao động', 'tiền lương', 'nghỉ phép'],
    answer: 'Bộ luật Lao động 2019 bảo vệ quyền và lợi ích hợp pháp của người lao động. Người sử dụng lao động phải ký hợp đồng lao động bằng văn bản nếu thời hạn từ 01 tháng trở lên.'
  },
  {
    keywords: ['đất đai', 'sổ đỏ', 'quyền sử dụng đất', 'thu hồi đất'],
    answer: 'Luật Đất đai 2013 quy định đất đai thuộc sở hữu toàn dân do Nhà nước đại diện chủ sở hữu. Người sử dụng đất được cấp Giấy chứng nhận quyền sử dụng đất (sổ đỏ).'
  },
  {
    keywords: ['hiến pháp', 'quyền con người', 'cơ bản'],
    answer: 'Hiến pháp 2013 là đạo luật gốc, quy định về chế độ chính trị, kinh tế, văn hóa, xã hội, quốc phòng, an ninh và quyền cơ bản của công dân.'
  }
];

export const searchLaws = (query) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      if (!query || query.trim() === '') {
        resolve(MOCK_LAWS);
        return;
      }
      const lowerQuery = query.toLowerCase();
      const results = MOCK_LAWS.filter(law => 
        law.title.toLowerCase().includes(lowerQuery) || 
        law.content.toLowerCase().includes(lowerQuery) ||
        law.type.toLowerCase().includes(lowerQuery)
      );
      resolve(results);
    }, 400);
  });
};

export const askQuestion = (question) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const lowerQ = question.toLowerCase();
      const match = MOCK_QA.find(qa => 
        qa.keywords.some(k => lowerQ.includes(k))
      );
      if (match) {
        resolve(match.answer);
      } else {
        resolve('Xin lỗi, tôi chưa có thông tin cụ thể về câu hỏi này. Vui lòng thử hỏi về: Hình sự, Dân sự, Lao động, Đất đai, hoặc Hiến pháp.');
      }
    }, 600);
  });
};

export const getLawSummary = (lawId) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(MOCK_SUMMARIES[lawId] || 'Chưa có bản tóm tắt cho văn bản này.');
    }, 300);
  });
};

export const getAllLaws = () => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(MOCK_LAWS), 200);
  });
};