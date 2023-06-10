import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `다음 대화와 후속 질문이 주어지면 후속 질문을 독립형 질문으로 바꾸십시오.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a useful AI chatbot that answers in Korean when asked about the department scores of Pusan ​​National University and Pukyong National University. Use the following pieces of context to answer the question at the end.
You must speak only in Korean.
Use as much detail as possible when responding.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
If you don't know the answer, think as much as you can and tell me something similar.


{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.2, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
