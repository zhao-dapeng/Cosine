
class NaiveBayesian:

    def __init__(self,ratio = 0.00001):
        self.ratio = ratio

    def LoadModel(self,modelfile = ''):
        self.type_probability = {}
        self.type_tag_probability = {}
        file = open(modelfile,'r',encoding='utf8')
        for line in file:
            datas = line.strip('\n').split(';')
            typeDatas = datas[0].split(':')
            self.type_probability[typeDatas[0]] = float(typeDatas[1])
            self.type_tag_probability[typeDatas[0]] = {}
            typeWordsDatas = datas[1].split(',')
            for typeWords in typeWordsDatas:
                words = typeWords.split(':')
                self.type_tag_probability[typeDatas[0]][words[0]] = float(words[1])

    def Predict(self,tags):
        maxValue = 0.0
        result = ''
        for type in self.type_probability:
            predictedValue = self.type_probability[type]
            str_test=type + ','+str(self.type_probability[type])+','
            for tag in tags:
                if tag in self.type_tag_probability[type]:
                    str_test+=tag+','+str(self.type_tag_probability[type][tag])+','
                    predictedValue *= self.type_tag_probability[type][tag]
                else:
                    predictedValue *= 0.00001
            if maxValue < predictedValue:
                maxValue = predictedValue
                result = type
        if result == '孕产' or result == '育儿':
            return '孕产/育儿'
        if result == '财经':
            return '财经/金融'
        if result == '时尚':
            return '时尚/美妆'
        if result == '故事' or result == '小说':
            return '故事/小说'
        if result == '健康':
            return '健康/养生'
        if result == '体育':
            return '体育/运动'
        return result






