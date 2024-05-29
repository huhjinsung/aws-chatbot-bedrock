# aws-chatbot-bedrock

## 데모 개요

- 보험 상담원은 고객들의 보험 상품, 가입, 보장, 절차 등과 관련된 사항들을 보험 약관을 기반으로 답변합니다. 그러나 보험 상품이 여러 개이고, 보험 약관 내용이 많은 관계로 모든 내용을 완벽히 숙지하기는 현실적으로 어려운 일입니다. 이러한 문제를 해결하기 위해 각 보험 상품별 보험 약관에 대한 답변을 제공하는 보험 약관 챗봇을 구현합니다. 보험 약관 PDF를 기반으로 RAG를 구성하며, RAG 구성을 빠르고 편하게 구성할 수 있는 Bedrock 기반의 Knowledge Base 및 Agent를 사용합니다.

- **챗봇 URL** : http://54.190.118.1:8501/

- **RAG에 사용한 문서** : https://static.kakaoinsure.com/notilus/files/20240401_해외여행보험_보험약관.pdf

## 데모 아키텍처
<img src="/architecture.png"></img>

1. Amazon Bedrock Knowledge Base 기능을 통해 Amazon S3에 저장된 보험 약관 PDF를 Embedding하여 Amazon OpenSearch Serverless에 저장합니다. Embedding을 위해 아래의 조건을 활용했습니다.
    1. **Embedding Model** : Titan Embedding G1
    2. **Chunking Strategy** : Fixed-Sized
        1. Max tokens : 300
        2. Overlap : 20%
2. 사용자는 Web 챗봇 시스템에서 질문을 남깁니다.
3. Amazon Bedrock Agent는 Knowledge Base에서 질문과 관련된 컨텍스트를 조회합니다. OpenSearch에 컨텍스트를 조회 할 때 Hybrid Search 기반으로 문서를 검색합니다. API를 호출하기 위한 문서는 아래 링크를 확인합니다.
    1. https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate.html
4. (Optional) Amazon Bedrock Agent를 활용하면 Knowledge Base 외에도 추가 Lambda를 활용하여 외부 API를 통해 관련된 정보를 갖고 올 수 있습니다. 이번 데모에서는 적용하지 않습니다.
5. Knowledge Base에서 조회한 컨테스트 기반으로 답변을 생성하고 사용자에게 전달합니다.

## 데모 결과
<img src="/demo_result.png"></img>

## 코드
먼저 코드에 사용할 라이브러리를 설치합니다.

```bash
python3 -m venv bedrock
source bedrock/bin/activate
pip install boto3 streamlit
```

아래의 파이썬 코드를 기반으로 bedrock.py 파일을 생성합니다. 아래 코드 중 <BedrockKnowledgeBase ID>와 <Your Region>을 각 사용자 환경에 맞춰 변경합니다.

```python
import time
import boto3
import streamlit as st

st.set_page_config(
    page_title="Kakaopay_Insurance",
    layout="centered"
)
       
# Setup bedrock
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name="us-west-2"
)

col11, col12, col13 = st.columns([1.5, 1, 3])

with col12:
    st.image("./kakaopay_insurance.png", width=300)

st.markdown("""
<div style="text-align:center;">
<h2>보험 상담원을 위한 보험 약관 챗봇</h2>
</div>
""", unsafe_allow_html=True)

def session_state() :
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    return st.session_state

submitted = False

with st.expander("**자주 하는 질문 리스트(FAQ)**   📌"):
    option = st.selectbox("FAQ 리스트 중 질문을 선택해주세요   ✅",
        ("",
            "청약 철회는 어떻게 하나요?",
            "보험료 환급은 어떻게 이뤄지나요?",
            "물건을 잃어버리는 사항도 보장이 되나요?",
            "보험금을 지급하지 않는 경우는 어떠한 사항이 있나요?"
            )
        )
    
    col1, col2, col3 = st.columns([5.5, 1, 1])
            
    with col3:
        send = st.button('Send')
        
    if send :
        submitted=True
        
session_state()

if submitted == True :

    if prompt := option :
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            response = bedrock_agent_runtime.retrieve_and_generate(
                input={
                    'text': prompt,
                },
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': <BedrockKnowledgeBase ID>,
                    'modelArn': 'arn:aws:bedrock:<YOUR-Region>::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0',
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 100,
                            'overrideSearchType': 'HYBRID'}
                        }
                    }
                }
            )
            
            answer = response['output']['text']
            context = response['citations'][0]['retrievedReferences'][0]['content']['text']
            reference = response['citations'][0]['retrievedReferences'][0]['location']['s3Location']['uri']

            response = f"\n\n {answer}\n\n\n **답변 생성을 위해 아래 컨텍스트를 활용했습니다.**\n\n {context}\n\n\n **문서 출처** : {reference}"
    
            # Simulate stream of response with milliseconds delay
            for chunk in response.split(' '): # fix for https://github.com/streamlit/streamlit/issues/868
                full_response += chunk + ' '
                if chunk.endswith('\n'):
                    full_response += ' '
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
    
            message_placeholder.markdown(full_response)
    
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    submitted = False

if submitted == False :
    if prompt := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            response = bedrock_agent_runtime.retrieve_and_generate(
                input={
                    'text': prompt,
                },
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': <BedrockKnowledgeBase ID>,
                    'modelArn': 'arn:aws:bedrock:<YOUR REGION>::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0',
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 100,
                            'overrideSearchType': 'HYBRID'}
                        }
                    }
                }
            )
            
            answer = response['output']['text']
            context = response['citations'][0]['retrievedReferences'][0]['content']['text']
            reference = response['citations'][0]['retrievedReferences'][0]['location']['s3Location']['uri']

            response = f"{answer}\n\n\n **답변 생성을 위해 아래 컨텍스트를 활용했습니다.**\n\n {context}\n\n\n **문서 출처** : {reference}"

            for chunk in response.split(' '): 
                full_response += chunk + ' '
                if chunk.endswith('\n'):
                    full_response += ' '
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
    
            message_placeholder.markdown(full_response)
    
        st.session_state.messages.append({"role": "assistant", "content": full_response})

```

아래의 명령어를 실행하고 출력되는 주소로 Web 서버로 이동합니다.