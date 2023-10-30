#include "tab1db.h"
#include "ui_tab1db.h"

void Tab1DB::connectToDatabase() {
    QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
    db.setHostName("3.39.54.145");
    db.setDatabaseName("Mart");
    db.setUserName("root");
    db.setPassword("971226");

    if (!db.open()) {
        qDebug() << "Database connection failed: " << db.lastError();
    } else {
        qDebug() << "Database connection successful!";
    }
}

Tab1DB::Tab1DB(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Tab1DB)
{
    ui->setupUi(this);
    pQTimer = new QTimer(this);
    listWidget = new QListWidget(this);
    connect(ui->pPBappQuit,SIGNAL(clicked()), qApp, SLOT(quit()));
    connect(ui->pCBkey1, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(1, int)));
    connect(ui->pCBkey2, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(2, int)));
    connect(ui->pCBkey3, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(3, int)));
    connect(ui->pCBkey4, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(4, int)));
    connect(ui->pCBkey5, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(5, int)));
    connect(ui->pCBkey6, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(6, int)));
    connect(ui->pCBkey7, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(7, int)));
    connect(ui->pCBkey8, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(8, int)));
    connect(ui->pCBkey9, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(9, int)));
    connect(ui->pCBkey10, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(10, int)));
    connect(ui->pCBkey11, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(11, int)));
    connect(ui->pCBkey12, SIGNAL(stateChanged(int)), this, SLOT(on_pCBkey_stateChanged(12, int)));



}
void Tab1DB::on_pCBkey_stateChanged(int index, int arg1) {
    QCheckBox* checkBox = nullptr;

    // 인덱스에 따라 해당 체크 박스를 선택
    switch (index) {
        case 1:
            checkBox = ui->pCBkey1;
            break;
        case 2:
            checkBox = ui->pCBkey2;
            break;
        case 3:
            checkBox = ui->pCBkey3;
            break;
        case 4:
            checkBox = ui->pCBkey4;
            break;
        case 5:
            checkBox = ui->pCBkey5;
            break;
        case 6:
            checkBox = ui->pCBkey6;
            break;
        case 7:
            checkBox = ui->pCBkey7;
            break;
        case 8:
            checkBox = ui->pCBkey8;
            break;
        case 9:
            checkBox = ui->pCBkey9;
            break;
        case 10:
            checkBox = ui->pCBkey10;
            break;
        case 11:
            checkBox = ui->pCBkey11;
            break;
        case 12:
            checkBox = ui->pCBkey12;
            break;
        default:
            break;
    }

    if (checkBox == nullptr) {
        return;
    }

    QString text = checkBox->text();

    if(arg1==2){

        ui->textEdit->append(text);
        shoppinglist.append(text);
    }
    else if(arg1==0)
    {
        shoppinglist.removeOne(text);

        QTextCursor cursor(ui->textEdit->document());

        // 텍스트를 찾아서 선택
        while (!cursor.atEnd()) {
            cursor = ui->textEdit->document()->find(text, cursor);

            if (!cursor.isNull()) {
                cursor.removeSelectedText();
            } else {
                break;
            }
        }
    }
}

// 각 체크 박스의 stateChanged 이벤트에서 on_pCBkey_stateChanged 호출
void Tab1DB::on_pCBkey1_stateChanged(int arg1) {
    on_pCBkey_stateChanged(1, arg1);
}void Tab1DB::on_pCBkey2_stateChanged(int arg1) {
    on_pCBkey_stateChanged(2, arg1);
}void Tab1DB::on_pCBkey3_stateChanged(int arg1) {
    on_pCBkey_stateChanged(3, arg1);
}void Tab1DB::on_pCBkey4_stateChanged(int arg1) {
    on_pCBkey_stateChanged(4, arg1);
}void Tab1DB::on_pCBkey5_stateChanged(int arg1) {
    on_pCBkey_stateChanged(5, arg1);
}void Tab1DB::on_pCBkey6_stateChanged(int arg1) {
    on_pCBkey_stateChanged(6, arg1);
}void Tab1DB::on_pCBkey7_stateChanged(int arg1) {
    on_pCBkey_stateChanged(7, arg1);
}void Tab1DB::on_pCBkey8_stateChanged(int arg1) {
    on_pCBkey_stateChanged(8, arg1);
}void Tab1DB::on_pCBkey9_stateChanged(int arg1) {
    on_pCBkey_stateChanged(9, arg1);
}void Tab1DB::on_pCBkey10_stateChanged(int arg1) {
    on_pCBkey_stateChanged(10, arg1);
}void Tab1DB::on_pCBkey11_stateChanged(int arg1) {
    on_pCBkey_stateChanged(11, arg1);
}void Tab1DB::on_pCBkey12_stateChanged(int arg1) {
    on_pCBkey_stateChanged(12, arg1);
}

// 나머지 체크 박스들도 유사한 방식으로 처리

void Tab1DB::on_pPBClear_clicked()
{
    ui->textEdit->clear();
    ui->pCBkey1->setChecked(false);
    ui->pCBkey2->setChecked(false);
    ui->pCBkey3->setChecked(false);
    ui->pCBkey4->setChecked(false);
    ui->pCBkey5->setChecked(false);
    ui->pCBkey6->setChecked(false);
    ui->pCBkey7->setChecked(false);
    ui->pCBkey8->setChecked(false);
    ui->pCBkey9->setChecked(false);
    ui->pCBkey10->setChecked(false);
    ui->pCBkey11->setChecked(false);
    ui->pCBkey12->setChecked(false);
}

void Tab1DB::on_pPBStart_clicked()
{
    connectToDatabase();
    // 데이터베이스 연결 확인
    if (!db.isOpen()) {
        qDebug() << "Database is not open. Cannot execute SQL command.";
        return;
    }

    QSqlQuery query(db);

    // 트랜잭션 시작
    query.exec("START TRANSACTION;");

    // Cart 테이블 내용 삭제
    query.exec("SET SQL_SAFE_UPDATES = 0;");
    query.exec("DELETE FROM Cart;");
    query.exec("SET SQL_SAFE_UPDATES = 1;");

    // shoppinglist에 있는 각 상품을 Cart 테이블에 삽입
    for (const QString &product : shoppinglist) {
        query.prepare("INSERT INTO Cart (product_name) VALUES (:product_name);");
        query.bindValue(":product_name", product);
        if (!query.exec()) {
            qDebug() << "Failed to insert into Cart: " << query.lastError();
        }
    }

    // 트랜잭션 커밋
    query.exec("COMMIT;");
}

Tab1DB::~Tab1DB()
{
    delete ui;
}


