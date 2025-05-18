from ultralytics import YOLO
import torch
import torch.nn as nn
import os
import yaml
import cv2
import numpy as np

class CustomYOLO:
    def __init__(self, model_type='yolov11s', weights_path=r'/home/tiago/yolo_project/weights/best', config_path=None):
        """
        Inicializa um modelo YOLO personalizado
        Args:
            model_type: Tipo de modelo base ('yolov8n', 'yolov8s', etc)
            weights_path: Caminho para os pesos personalizados (.pt, .pth)
            config_path: Caminho para arquivo de configuração YAML (opcional)
        """
        # Criar/carregar modelo
        if weights_path and os.path.exists(weights_path):
            print(f"Carregando pesos personalizados de: {weights_path}")
            
            # Opção 1: Se os pesos forem compatíveis com Ultralytics
            try:
                self.model = YOLO(weights_path)
                print("Pesos carregados com sucesso usando a API da Ultralytics")
            
            # Opção 2: Carregar apenas os pesos (state_dict) para modelo pré-criado
            except Exception as e:
                print(f"Erro ao carregar diretamente: {e}")
                print("Tentando método alternativo de carregamento...")
                
                self.model = YOLO(model_type)
                state_dict = torch.load(weights_path, map_location='cpu')
                
                # Verificar se os pesos estão em formato de state_dict ou envoltos em alguma estrutura
                if isinstance(state_dict, dict):
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                
                # Tentar carregar parâmetros compatíveis
                try:
                    self.model.model.load_state_dict(state_dict, strict=False)
                    print("Pesos carregados com correspondência parcial")
                except Exception as e:
                    print(f"Erro ao carregar pesos: {e}")
                    print("Verifique se a arquitetura do modelo corresponde aos pesos")
        else:
            # Criar modelo padrão
            print(f"Criando modelo {model_type} padrão")
            self.model = YOLO(model_type)
        
        # Carregar configurações personalizadas se fornecidas
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Configurações carregadas de: {config_path}")
        
    def predict(self, image_path, conf=0.25):
        """Realiza inferência em uma imagem"""
        return self.model.predict(image_path, conf=conf)
    
    def train(self, data_yaml, epochs=100, imgsz=640):
        """Treina ou fine-tuna o modelo"""
        self.model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
    
    def export(self, format='onnx'):
        """Exporta o modelo para diferentes formatos"""
        self.model.export(format=format)

def create_visualization_with_side_labels(image_path, results, output_path='resultado_legend.jpg'):
    """
    Cria uma visualização com as bounding boxes na imagem e labels na lateral
    """
    # Carregar a imagem original
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extrair detecções
    boxes = results[0].boxes
    classes = results[0].names
    
    # Determinar cores únicas para cada classe
    unique_classes = set(int(box.cls.item()) for box in boxes)
    colors = {cls_id: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
              for cls_id in unique_classes}
    
    # Criar uma imagem ampliada para acomodar a legenda
    legend_width = 200  # Largura da área de legenda
    height, width = img.shape[:2]
    combined_img = np.ones((height, width + legend_width, 3), dtype=np.uint8) * 255
    combined_img[:, :width] = img
    
    # Desenhar as bounding boxes na imagem (sem labels)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls.item())
        color = colors[cls_id]
        
        # Desenhar retângulo sem texto
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), color, 2)
    
    # Desenhar legenda no painel lateral
    y_offset = 30
    for cls_id in unique_classes:
        # Obter a contagem desta classe
        count = sum(1 for box in boxes if int(box.cls.item()) == cls_id)
        
        # Obter nome e cor da classe
        class_name = classes[cls_id]
        color = colors[cls_id]
        
        # Desenhar quadrado colorido e nome da classe
        cv2.rectangle(combined_img, (width + 20, y_offset - 15), (width + 40, y_offset + 5), color, -1)
        cv2.putText(combined_img, f"{class_name} ({count})", (width + 45, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        y_offset += 30
    
    # Salvar a visualização
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_img)
    print(f"Imagem com legenda lateral salva como '{output_path}'")
    
    return combined_img

def main():
    # Caminho para os pesos
    weights_path = '/home/tiago/Downloads/weights.pt'
    
    if not os.path.exists(weights_path):
        print(f"ERRO: Arquivo de pesos não encontrado em {weights_path}")
        return
    
    # Instancia o modelo com os pesos
    model = CustomYOLO(
        model_type='yolov8n',  # Alterado para um modelo padrão válido
        weights_path=weights_path
    )
    
    # Caminho para uma imagem de teste
    image_path = r'/home/tiago/yolo_project/Dataset_VSI/test/images/5194_LP_MC_0181_png.rf.02a33f420b131ba2d404dcbaf23117f4.jpg'
    
    if not os.path.exists(image_path):
        print(f"ERRO: Imagem de teste não encontrada em {image_path}")
        return
    
    # Executa inferência na imagem
    results = model.predict(image_path, conf=0.25)
    
    # Criar visualização personalizada
    create_visualization_with_side_labels(image_path, results)
    
    # Salvar também a visualização padrão para comparação
    results[0].save(filename='resultado_padrao.jpg')
    
    # Exportar o modelo para formato ONNX
    print("Exportando modelo para formato ONNX...")
    model.export(format='onnx')
    print("Modelo exportado com sucesso!")
    
    # Mostrar estatísticas
    print(f"Detecções encontradas: {len(results[0].boxes)}")

if __name__ == "__main__":
    main()