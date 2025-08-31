import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)

print(df.head())
print("-------------------------------------------------")
print(df["Name"])
print("-------------------------------------------------")
print(df[df["Age"] >28])
print("-------------------------------------------------")
print(df.describe())
print("-------------------------------------------------")
print(df.groupby("Name")["Age"].mean())

# csv파일 불러오기(웹으로)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("-------------------------------------------------")
print("@@@@@@@@@count@@@@@@@@@@@@@@@@@ " , df.count())
print("-------------------------------------------------")
print(df.head())        # 위 5줄 확인
print("-------------------------------------------------")
print(df.info())        # 컬럼 정보, 결측치 확인
print("-------------------------------------------------")
print(df.describe())    # 숫자 컬럼 통계 정보 
print("-------------------------------------------------")
df['IsChild'] = df['Age'] < 20 # 나이가 18세 미만인 경우 True / Filed 생성
print(df[['Name', 'Age', 'IsChild']].head())
df['Age'] = df['Age'].fillna(df['Age'].mean()) #  빈 값을 평균값으로 채우기
print("-------------------------------------------------")
print(df['Age'].isnull().sum()) # 결측치 개수 확인