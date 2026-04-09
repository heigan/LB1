import subprocess
import os

def main():
    os.makedirs('reports', exist_ok=True)
    
    # Получаем DOT-представление графа
    result = subprocess.run(
        ['dvc', 'dag', '--dot'],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Сохраняем в файл
    with open('reports/dvc_graph.dot', 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    print("Граф сохранён: reports/dvc_graph.dot")
    
    # Попытка конвертации через graphviz (если установлен)
    try:
        subprocess.run(
            ['dot', '-Tpng', 'reports/dvc_graph.dot', '-o', 'reports/dvc_graph.png'],
            check=True
        )
        print("Изображение сохранено: reports/dvc_graph.png")
    except FileNotFoundError:
        print("⚠️ Graphviz не найден в PATH. Для конвертации в PNG установите Graphviz:")
        print("   1. Скачайте с https://graphviz.org/download/")
        print("   2. Добавьте C:\\Program Files\\Graphviz\\bin в системный PATH")
        print("   3. Перезапустите терминал")
        print("   Или используйте текстовый вывод: dvc dag --md")

if __name__ == '__main__':
    main()