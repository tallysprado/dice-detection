<p align="center">
 <a href="#objetivo">Objetivo</a> •
 <a href="#demonstracao">Demonstração</a> • 
 <a href="#filtros">Filtros</a>
</p>

### Detectando e contando os dados 
O arquivo example.py utiliza OpenCV para localizar os dados e reconhecer suas faces.

### Demonstração
Dentro do diretório execute
```
python3 example.py
```
![image](https://user-images.githubusercontent.com/19524848/115444523-0ba9dc80-a1eb-11eb-9223-360fbb905110.png)
![image](https://user-images.githubusercontent.com/19524848/115444560-195f6200-a1eb-11eb-944e-ce2669ca2cf7.png)
![image](https://user-images.githubusercontent.com/19524848/115444601-27ad7e00-a1eb-11eb-8f8e-54543f95c263.png)

### Filtros
- Blur removerá o ruído de brilho
- Threshold é utilizado para remover informações indesejadas
- Erode maximizará informações úteis
- _Blob detection_ irá localizar os pontos dos dados de acordo com seus parâmetros
- _Canny_ foi utilizado para auxiliar na detecção de contornos e extrair dados únicos

### Features 
- [X] Deteção de todas as imagens
- [X] Documentação
- [ ] Detecção utilizando apenas um dos dois métodos de segmentação escolhidos
