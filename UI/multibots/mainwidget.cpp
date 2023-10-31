#include "mainwidget.h"
#include "ui_mainwidget.h"

#include <QDebug>
#include <typeinfo>
#include <iostream>
MainWidget::MainWidget(int argc, char **argv, QWidget *parent)
    : QWidget(parent), ui(new Ui::MainWidget)
{
    ui->setupUi(this);
    setWindowTitle("Multibots");

    ui->tabWidget->setCurrentIndex(0);

    pTab1DB = new Tab1DB(ui->pTab1);
    ui->pTab1->setLayout(pTab1DB->layout());

    pTab2Cam = new Tab2Cam(ui->pTab2);
    ui->pTab2->setLayout(pTab2Cam->layout());

    pTab3SocketClient = new Tab3SocketClient(ui->pTab3);
    ui->pTab3->setLayout(pTab3SocketClient->layout());

    connect(pTab3SocketClient,SIGNAL(goGoalSig(double, double, double, double)),pTab2Cam,SLOT(goal_PubSlot(double, double, double, double)));
}

MainWidget::~MainWidget()
{
    delete ui;
}
