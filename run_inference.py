from VSI import CustomYOLO, create_visualization_with_side_labels
import os

def main():
    # Caminho para os pesos (verifique se este arquivo existe)
    weights_path = '/home/tiago/Downloads/weights.pt'
    
    # Verifique se o arquivo de pesos existe
    if not os.path.exists(weights_path):
        print(f"ERRO: Arquivo de pesos não encontrado em {weights_path}")
        return
    
    # Instancia o modelo com os pesos
    model = CustomYOLO(
        model_type='yolov11s',  # Modelo base
        weights_path=weights_path  # Caminho para os pesos
    )
    
    # Caminho para uma imagem de teste
    image_path = r'/home/tiago/yolo_project/Dataset_VSI/test/images/5194_LP_MC_0181_png.rf.02a33f420b131ba2d404dcbaf23117f4.jpg'
    
    # Verifique se a imagem existe
    if not os.path.exists(image_path):
        print(f"ERRO: Imagem de teste não encontrada em {image_path}")
        return
    
    # Executa inferência na imagem
    results = model.predict(image_path, conf=0.25)
    
    # Gerar visualização personalizada com labels na lateral
    create_visualization_with_side_labels(image_path, results, output_path='resultado_lateral.jpg')
    
    # Exportar o modelo para formato ONNX
    print("Exportando modelo para formato ONNX...")
    model.export(format='onnx')
    print("Modelo exportado com sucesso!")

    # Mostra os resultados
    print(f"Detecções encontradas: {len(results[0].boxes)}")
    
    # Também salvar a visualização padrão para comparação
    results[0].save(filename='resultado_padrao.jpg')
    print("Imagens salvas como 'resultado_lateral.jpg' e 'resultado_padrao.jpg'")

if __name__ == "__main__":
    main()